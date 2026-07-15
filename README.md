# RAG-Audit

MVP para auditar eventos de entrada em salas críticas. Ele recebe o identificador reconhecido pela câmera, recupera pessoa, permissão, escala, sala, chamados e políticas, classifica o contexto com regras auditáveis, grava o log, gera PDF e enfileira alertas.

O sistema **não controla a fechadura**. A API principal recebe um resultado de identidade (`user_id`) e metadados mínimos. Para a VIPC 1230 G2, que não possui reconhecimento embarcado, existe um worker local que reconhece quem passa pela porta com **InsightFace (RetinaFace + ArcFace)**, consenso em vários quadros e auto-aprendizado da base. Imagens e vetores biométricos não entram no webhook, nos PDFs ou no SQLite da API.

## O que já funciona

- Webhook autenticado e idempotente para eventos de câmera.
- Validação da sala contra o cadastro da câmera.
- Horários RFC 3339 obrigatoriamente acompanhados de offset/fuso.
- SQLite em modo WAL com dados simulados dos três cenários do desafio.
- Motor determinístico com decisões `AUTHORIZED`, `JUSTIFIED` e `ANOMALY`.
- Recuperação das evidências e políticas usadas na decisão.
- Dashboard protegido por HTTP Basic, filtros, métricas de SLA e atualização ao vivo.
- Relatório PDF individual e consolidado conforme os filtros.
- Outbox persistente com reenvio de alertas sem bloquear o webhook.
- Testes automatizados de regras, segurança, idempotência, fuso e PDF.

## Arquitetura

```text
Câmera / controle de acesso
  └─ POST autenticado
       └─ validação + idempotência
            ├─ SQLite: pessoa, sala, escala, chamado, políticas
            ├─ motor determinístico de risco
            ├─ log + contexto + versão da política
            ├─ outbox assíncrona → webhook de alerta
            └─ dashboard / API / PDF
```

A implementação usa uma forma local e auditável de recuperação + geração: busca fatos estruturados e políticas aplicáveis e gera o texto em português por template. Um LLM pode ser acrescentado depois apenas para redação; a classificação, as fontes e o alerta devem continuar determinísticos para não alucinar justificativas nem comprometer o SLA de 3 segundos.

A API de auditoria corresponde à **fase 1 do MVP**. A fundação da fase de visão já está implementada, mas ainda precisa ser comissionada e calibrada na câmera real. Ainda não há busca vetorial/semântica, LLM, prova de vida, receptor de e-mail Intelbras nem adaptador específico de Teams/Slack/Telegram. Chamar o protótipo atual de “RAG com banco vetorial” seria incorreto.

O desenho específico para a câmera informada no piloto está em
[Integração da Intelbras VIPC 1230 G2](docs/integracao-vipc-1230-g2.md). Ele define o critério
visual de entrada, a sincronização possível da base Intelbras e os limites da evidência sem sensor
da porta.

O roteiro completo para levar o projeto à rede real, incluindo o novo gatilho por e-mail com foto,
está em [Plano de implantação no ambiente Intelbras](docs/plano-implantacao-ambiente-intelbras.md).

## Reconhecimento facial na porta (fase de visão)

Esta é a evolução implementada e em operação: o worker local reconhece **quem** passa pela porta comparando o rosto ao vivo com a base de fotos do controle de acesso — sem controlar a fechadura e sem enviar nenhuma imagem/vetor biométrico à API.

### Tecnologia e bibliotecas

- **InsightFace** (pacote `buffalo_l`), em CPU via **ONNX Runtime**:
  - **RetinaFace** para detectar rostos.
  - **ArcFace (ResNet-50, embedding de 512 dimensões)** para a "impressão digital" do rosto.
- **OpenCV** para captura RTSP e imagem; **NumPy** para os vetores.
- Similaridade por **cosseno**. O ArcFace separa muito bem as identidades (duas pessoas diferentes ficam perto de 0), o que permite um limiar baixo (~0,38) para **vencer o ângulo** da câmera de teto sem confundir pessoas. O YuNet/SFace do OpenCV foi a base inicial; a migração para ArcFace elevou a precisão e a robustez a ângulo.

### Como funciona (fluxo do worker)

```text
RTSP (DVR Intelbras) → RetinaFace detecta os rostos do quadro
  → ArcFace gera o embedding de cada rosto
  → compara com a base (pessoas importadas do controle de acesso)
  → consenso em vários quadros + margem entre 1º e 2º candidato
  → regra de porta: só conta quem CHEGOU (se moveu na zona da porta)
  → registra uma vez por visita (presença) → evento SEM imagem para a API
  → salva a melhor foto (privada) + auto-aprende o ângulo da pessoa
```

### Modelo "reconhecido na porta"

Como a porta tem fechadura eletrônica (quem não é autorizado nem abre), o momento de registrar é o **reconhecimento na porta** — não uma travessia direcional. O evento usa `entry_evidence=VISION_FACE_AT_DOOR` e `door_result=NOT_REPORTED` (a câmera não prova a liberação da fechadura). A contagem é precisa por três regras:

- **Uma pessoa = um registro por visita** (presença): enquanto continua aparecendo não registra de novo; só reconta após ~5 s de ausência (saiu e voltou).
- **Ignora quem está parado** (ex.: alguém sentado na direção da porta): só conta o rosto que **se move** ao entrar; fixos no enquadramento não viram registro.
- **Múltiplas pessoas / carona**: cada pessoa é reconhecida e registrada separadamente, mesmo passando uma atrás da outra (o worker detecta a troca de identidade na trilha).

### Auto-aprendizado da base

Quando reconhece alguém com confiança (mas em um ângulo ainda não bem coberto), o worker guarda **só o embedding** (~2 KB) daquele rosto como nova referência da pessoa, nos ângulos reais da porta. A base fica mais precisa com o uso, **sem cadastrar rosto a rosto**. Travas de segurança: só aprende match confiante (0,50–0,85), qualidade mínima, no máximo 5 referências por pessoa, tudo auditável em `data/private/gallery/learned.db`. As referências aprendidas carregam instantâneas no restart (o conjunto base usa um cache de embeddings).

### Evidência (foto) e privacidade

A imagem **nunca** trafega no webhook nem entra no SQLite/PDF da auditoria. O worker salva, em armazenamento privado, a **cena completa** e um **recorte do rosto** (arquivos nomeados por hash), e o evento carrega apenas uma **referência opaca**. O painel exibe a foto atrás da autenticação de administrador.

### Configurar e rodar a visão

```powershell
python -m pip install -e ".[vision]"   # numpy, opencv, onnxruntime, insightface
# os modelos ArcFace/RetinaFace (buffalo_l) são baixados na primeira execução

# importar a base de fotos do controle de acesso (ex.: ZIP do Intelbras InControl)
python scripts\import_incontrol_gallery.py <origem> data\private\gallery
python scripts\sync_gallery_people.py       # sincroniza nomes/IDs no banco de auditoria

# testar câmera/RTSP e validar toda a cadeia
python scripts\probe_intelbras_camera.py
python scripts\run_vision_worker.py --check

# operar (grava eventos com foto no painel; use RAG_AUDIT_VISION_DRY_RUN=false)
python scripts\run_vision_worker.py
```

Limiar, qualidade, fps e parâmetros de aprendizado ficam no `.env` (veja `.env.example`). A base biométrica é dado pessoal sensível: trate `data/` como privado (ele fica fora do Git).

## Executar no Windows

Requer Python 3.11 ou superior.

```powershell
cd C:\caminho\para\rag-audit
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

Abra `http://127.0.0.1:8000/dashboard`. As credenciais exclusivamente para desenvolvimento são:

- usuário: `admin`
- senha: `change-me`

Em outro terminal, envie os três eventos do PDF:

```powershell
cd C:\caminho\para\rag-audit
.\.venv\Scripts\Activate.ps1
python scripts\send_demo_events.py
```

O resultado esperado é:

```text
Lucas:   AUTHORIZED / LOW
Mariana: JUSTIFIED  / MEDIUM (incidente #402)
Roberto: ANOMALY    / HIGH (alerta para revisão)
```

Repetir o script demonstra a idempotência: o servidor devolve o mesmo recibo sem duplicar log ou alerta.

## Executar com Docker

Primeiro crie e edite o arquivo de ambiente, principalmente as credenciais:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

O diretório local `./data` é montado em `/app/data` e preserva o SQLite. O MVP deve rodar com **um worker**, pois o stream em memória e a escrita SQLite foram desenhados para uma única instância.

## Contrato da câmera

Endpoint: `POST /v1/webhooks/access-events`

Cabeçalho obrigatório: `X-Camera-Key`

```json
{
  "event_id": "vendor-event-0001",
  "camera_id": "cam-ti-01",
  "user_id": "EMP001",
  "room_id": "sala_ti_01",
  "timestamp": "2026-07-14T14:00:00-03:00",
  "door_result": "GRANTED",
  "recognition_confidence": 0.98
}
```

Regras do contrato:

- `event_id` identifica o evento no equipamento e impede duplicação.
- `timestamp` sem `-03:00`, outro offset ou `Z` é rejeitado; o servidor não adivinha o fuso.
- `room_id` precisa coincidir com a sala vinculada a `camera_id` no banco.
- `door_result` aceita `GRANTED`, `DENIED` ou `NOT_REPORTED`. A classificação contextual nunca é apresentada como resultado físico da porta.
- `recognition_confidence` é opcional, varia de 0 a 1 e não autoriza a entrada. Abaixo do limiar cadastrado, apenas gera alerta crítico para revisão.
- O mesmo `event_id` e conteúdo retorna `200`; o mesmo ID com conteúdo diferente retorna `409`.
- A resposta da câmera é um recibo mínimo com decisão, risco, motivos e ID do alerta; nomes, cargos, escalas, chamados e documentos de política ficam restritos à API administrativa.

Quando a câmera oferece RTSP e não envia `user_id`, o worker opcional executa detecção facial, tracking, comparação e confirmação direcional fora da API. Ele não fornece prova de vida. A foto enviada por e-mail será integrada como gatilho/observação complementar após analisarmos o MIME real do firmware; uma foto isolada não produz entrada confirmada.

## Regras de classificação

| Resultado | Risco | Condições principais |
|---|---:|---|
| `AUTHORIZED` | baixo | pessoa ativa, permissão para a sala e horário dentro da escala |
| `JUSTIFIED` | médio | pessoa ativa e autorizada, fora da escala, mas com incidente P1/P2 ativo, atribuído a ela e ligado à sala |
| `ANOMALY` | alto | ausência de permissão ou acesso fora da escala sem incidente qualificável |
| `ANOMALY` | crítico | pessoa desconhecida/inativa ou confiança abaixo do limiar |

Fim de semana não é anomalia por si só: uma escala válida no domingo continua sendo normal. Um chamado de outra pessoa, outra sala, baixa prioridade, fechado ou aberto somente após o evento não justifica o acesso.

Os alertas dizem “requer verificação humana”; não acusam a pessoa. Nenhuma sanção ou decisão trabalhista deve decorrer apenas desta classificação.

## API administrativa

As rotas abaixo exigem HTTP Basic:

| Método e rota | Uso |
|---|---|
| `GET /v1/access-events` | listar e filtrar logs |
| `GET /v1/access-events/{event_id}` | ver evento, fontes e contexto completos |
| `GET /v1/access-events/stream` | atualização ao vivo por SSE |
| `GET /v1/metrics` | totais, SLA, média e p95 |
| `GET /v1/rooms` | salas disponíveis |
| `GET /v1/access-events/{event_id}/report.pdf` | PDF individual |
| `GET /v1/reports/access-events.pdf` | PDF consolidado filtrado |
| `GET /docs` | documentação interativa protegida pela mesma autenticação |

Filtros de listagem, métricas e PDF: `from`, `to`, `room_id`, `user_id`, `decision`, `risk_level`, `alert_status` e `q`.

`processing_ms` e o cartão de SLA da API medem do recebimento do webhook até a persistência da decisão/outbox. `ingestion_delay_ms` mede câmera → API e `decision_e2e_ms` mede câmera → decisão. A API de métricas expõe percentuais separados (`api_sla_percentage` e `e2e_sla_percentage`); o critério “após a entrada” deve usar o segundo e depende de relógios sincronizados.

## Alertas

Defina `RAG_AUDIT_ALERT_WEBHOOK_URL`, `RAG_AUDIT_ALERT_CHANNEL` e `RAG_AUDIT_PUBLIC_BASE_URL` no `.env`. O envio usa timeout curto e o cabeçalho de idempotência com o ID do evento. Por padrão, o payload usa apenas o ID pseudônimo e o contexto mínimo; `RAG_AUDIT_ALERT_INCLUDE_PERSONAL_DATA=true` inclui nome, cargo, narrativa e confiança somente quando houver aprovação para expor esses dados ao operador do canal.

Sem URL configurada, o alerta fica registrado como `NOT_CONFIGURED` na outbox e aparece no dashboard. Se uma URL for configurada e o serviço reiniciado, esses alertas voltam à fila. Com falha temporária, passam por `RETRYING`; depois do limite configurado, ficam `FAILED`. O log do evento permanece intacto em todos os casos.

Teams, Slack e Telegram possuem formatos próprios. Em produção, use um pequeno adaptador por canal ou aponte para uma automação interna que aceite o JSON documentado na resposta do webhook.

## Testes

```powershell
pytest
```

Os testes usam um banco temporário e não alteram `data/rag_audit.db`.

## Estrutura do projeto

```text
app/
  alerts.py        worker da outbox
  config.py        ambiente e credenciais
  database.py      esquema, dados de demonstração e consultas
  main.py          API, autenticação, dashboard e stream
  models.py        contrato e validação
  reports.py       PDFs individual e consolidado
  risk_engine.py   regras e narrativa explicável
  vision/          RTSP, galeria, reconhecimento, tracking e outbox visual
  static/          CSS e JavaScript sem build
  templates/       dashboard Jinja
scripts/
  download_vision_models.py
  import_incontrol_gallery.py
  probe_intelbras_camera.py
  run_vision_worker.py
  sync_gallery_people.py
  send_demo_events.py
docs/
  integracao-vipc-1230-g2.md
  plano-implantacao-ambiente-intelbras.md
tests/
```

## Segurança, privacidade e produção

Biometria vinculada a uma pessoa é dado pessoal sensível na LGPD. Antes de um piloto real, o controlador deve envolver jurídico e encarregado/DPO, definir e documentar a hipótese legal aplicável, fazer avaliação de necessidade e riscos, informar as pessoas afetadas e estabelecer retenção e descarte. Consulte o texto oficial da [LGPD, especialmente os arts. 5º, 11 e 20](https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709compilado.htm) e os materiais da [Autoridade Nacional de Proteção de Dados](https://www.gov.br/anpd/pt-br).

Este repositório é um MVP técnico, não um parecer jurídico e não comprova conformidade ou certificação ISO 27001. Para produção ainda são necessários, no mínimo:

- TLS/mTLS ou rede privada entre câmera e API, segredo único por dispositivo e proteção contra replay;
- autenticação corporativa com RBAC/MFA no lugar de HTTP Basic;
- criptografia do volume e backups; SQLite puro não cifra o arquivo;
- trilha administrativa imutável, política de retenção e exclusão automática;
- revisão humana e canal de contestação de falso positivo;
- testes do reconhecimento facial, prova de vida, limiar e desempenho por condições reais;
- filas/worker externos e banco de produção se houver múltiplas instâncias;
- adaptação e teste do canal real de alerta;
- cadastro/importação administrativa de pessoas, câmeras, permissões, escalas, chamados e políticas (o MVP só traz fixtures de demonstração);
- monitoramento, resposta a incidentes e validação formal do SLA.

Em produção, configure obrigatoriamente `RAG_AUDIT_SEED_DEMO_DATA=false` e `RAG_AUDIT_ENFORCE_EVENT_FRESHNESS=true`. O startup rejeita dados de demonstração, credenciais fracas e configurações numéricas inválidas quando `RAG_AUDIT_ENV=production`.
