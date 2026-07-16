# RAG-Audit

Aplicação para auditar entradas observadas em salas críticas. A API recebe um evento normalizado, consulta pessoa, permissão, escala, sala, chamados e políticas, classifica o contexto com regras determinísticas, registra a trilha de auditoria, gera PDF e envia alertas por uma fila persistente.

O sistema não controla a fechadura. Na sala do piloto, a porta abre por senha, mas esse teclado não fornece log para a aplicação. A câmera também não comprova que uma senha foi aceita. Por isso, os relatórios distinguem uma entrada observada visualmente de uma liberação física da porta.

## Estado do projeto

A API, o dashboard, os relatórios e o contrato de eventos podem ser executados em laboratório. A fase de visão foi estruturada para a câmera Intelbras VIPC 1230 G2, mas ainda precisa ser instalada, medida e calibrada na rede real. Este repositório não deve ser apresentado como um sistema de reconhecimento já comissionado.

O que está coberto:

- webhook autenticado e idempotente;
- validação de câmera, sala, horário e fuso;
- regras auditáveis para `AUTHORIZED`, `JUSTIFIED` e `ANOMALY`;
- dashboard protegido, filtros, indicadores e atualização por SSE;
- relatórios PDF individuais e consolidados;
- outbox persistente para eventos visuais e alertas;
- importação defensiva de uma galeria autorizada;
- detecção, reconhecimento, tracking e consenso temporal;
- rastreamento direcional da porta para o interior;
- armazenamento privado de evidências com referência opaca;
- quarentena e revisão de referências faciais aprendidas;
- testes automatizados e CI.

Ainda dependem do ambiente:

- acesso RTSP à câmera e ajuste do enquadramento;
- identificação da origem oficial das fotos cadastradas na Intelbras;
- calibração de ROI, zonas, linha, qualidade e limiares;
- avaliação com pessoas autorizadas nas condições reais de luz e movimento;
- validação do e-mail com foto, caso ele seja usado como gatilho complementar;
- definição formal de retenção, revisão humana e operação.

## Arquitetura

```text
Intelbras VIPC 1230 G2
  RTSP
    |
    v
worker local de visão
  ROI -> SCRFD-10GF -> qualidade -> ArcFace -> tracking
      -> consenso temporal -> trajetória porta/linha/interior
      -> evidência privada + outbox offline
    |
    | evento normalizado, sem imagem ou vetor
    v
FastAPI RAG-Audit
  regras -> log -> alerta -> dashboard -> PDF
```

A API não processa o stream. Frames, recortes e embeddings ficam no lado privado do worker. O webhook recebe somente os metadados necessários à auditoria.

O motor de contexto não depende de LLM. Ele recupera fatos estruturados e redige a narrativa por template, mantendo a decisão e as fontes reproduzíveis. Um modelo de linguagem pode ser acrescentado no futuro para redação, mas não deve decidir autorização, risco ou justificativa.

## Reconhecimento facial

O worker usa o pacote InsightFace `buffalo_l`:

- `det_10g.onnx`: detector SCRFD-10GF;
- `w600k_r50.onnx`: ArcFace W600K R50, com embeddings de 512 dimensões;
- ONNX Runtime para inferência;
- OpenCV para RTSP e manipulação dos frames;
- similaridade por cosseno, com margem entre o primeiro e o segundo candidato.

Os pesos não fazem parte do repositório e nunca são baixados pelo aplicativo. O operador precisa obtê-los por uma fonte autorizada, conferir a licença aplicável ao uso pretendido e preparar um bundle externo. O manifesto registra os arquivos, tamanhos, hashes e a referência da autorização de uso. Na inicialização, o worker compara o SHA-256 autorizado com o conteúdo montado e recusa um bundle ausente ou alterado.

Estrutura esperada:

```text
data/vision-models/
  bundle-manifest.json
  models/
    buffalo_l/
      det_10g.onnx
      w600k_r50.onnx
```

Depois de colocar os arquivos obtidos legitimamente no diretório:

```powershell
python scripts\prepare_vision_models.py `
  --root data\vision-models `
  --license-reference "registro interno ou documento aplicável" `
  --accept-model-license

python scripts\verify_vision_models.py `
  --root data\vision-models `
  --expected-fingerprint <SHA-256 exibido no passo anterior>
```

O fingerprint precisa ser mantido em `RAG_AUDIT_VISION_MODEL_BUNDLE_SHA256`. Trocar qualquer peso exige nova preparação, avaliação e aprovação.

### Qualidade antes da identidade

O reconhecimento só deve comparar uma face quando a captura atende aos limites calibrados. Entre os sinais considerados estão:

- IED, a distância em pixels entre os olhos, para rejeitar rosto pequeno;
- nitidez, para reduzir erro por movimento ou foco;
- iluminação e contraste;
- pose, principalmente yaw, pitch e roll;
- detecção e alinhamento válidos;
- ausência de corte relevante ou oclusão.

Não existe valor universal para esses limites. A VIPC 1230 G2 é fixa, tem resolução e campo de visão limitados e não será substituída neste projeto. A precisão dependerá principalmente do posicionamento, da região útil da imagem, da luz e do tamanho do rosto na soleira. A ROI deve cobrir a zona da porta, a linha e a zona interna, com uma pequena margem; áreas da sala que não participam da entrada não precisam alimentar o detector.

### Consenso conservador

Um frame isolado não atribui identidade. O worker mantém a trilha do rosto e só aceita um nome quando:

- há observações suficientes no histórico;
- a mesma identidade vence em vários frames e também de forma consecutiva;
- a proporção de matches é suficiente, considerando frames desconhecidos, ambíguos e de baixa qualidade;
- a similaridade supera o limiar calibrado;
- a margem para o segundo candidato é suficiente;
- a qualidade permanece aceitável.

Mudança de aparência ou de identidade na mesma trilha reinicia o consenso. `LOW_QUALITY` é um estado interno do frame e também quebra a sequência recente; se não houver evidência suficiente, o evento sai como `UNKNOWN`, sem nome forçado.

## Regra de entrada

O modo padrão é `entry`: a mesma trilha precisa aparecer na zona da porta, cruzar a linha no sentido configurado e alcançar a zona interna dentro dos limites de tempo, deslocamento e permanência.

```text
zona da porta -> linha no sentido IN -> zona interna
```

As coordenadas são normalizadas e ficam em um arquivo privado derivado de `config/camera-calibration.example.json`. A linha possui zona morta e segmento finito para reduzir eventos por tremor, oscilação ou cruzamento fora da passagem. O carregamento rejeita polígonos auto-intersectantes, direção invertida e linha que não cubra o caminho entre as zonas.

O modo `door` registra somente um rosto observado perto da porta. Ele não comprova direção, senha ou abertura. Por segurança, deve ficar em dry-run; uma emissão para a API exige uma liberação operacional explícita e continua usando `door_result=NOT_REPORTED`.

Em ambos os casos:

- `VISION_LINE_CROSSING` significa entrada observada pela trajetória;
- `VISION_FACE_AT_DOOR` significa presença observada perto da porta;
- nenhum deles significa senha aceita;
- uma pessoa desconhecida ou ambígua não recebe um nome forçado;
- o resultado serve para auditoria e revisão humana.

## Galeria oficial

A galeria oficial deve vir de fotos exportadas por um caminho suportado do InControl, Defense IA ou controlador usado no ambiente. Templates proprietários não são intercambiáveis com fotos nem com embeddings ArcFace.

```powershell
python scripts\import_incontrol_gallery.py <origem> data\private\gallery
python scripts\sync_gallery_people.py `
  --database data\api\rag_audit.db `
  data\private\gallery\manifest.json
```

## Contexto de acesso

Importar a galeria cria ou atualiza pessoas, mas não concede acesso. Salas, câmeras, pessoas, permissões, escalas e políticas são provisionadas por um manifesto JSON privado. O exemplo versionado está em `config/access-context.example.json`; a cópia preenchida deve ficar em `data/private/config/access-context.json`, fora do Git, e não pode conter senha, chave, token ou URL RTSP.

```powershell
Copy-Item config\access-context.example.json `
  data\private\config\access-context.json

python scripts\provision_access_context.py `
  data\private\config\access-context.json `
  --database data\api\rag_audit.db `
  --dry-run `
  --replace-assignments

python scripts\provision_access_context.py `
  data\private\config\access-context.json `
  --database data\api\rag_audit.db `
  --replace-assignments
```

O dry-run valida referências, fusos, horários, IDs e o conjunto de políticas dentro de uma transação revertida. `--replace-assignments` remove permissões e escalas omitidas somente para as pessoas presentes no manifesto; sem essa opção, o comportamento é de merge. Uma política publicada é imutável: alterações exigem uma nova versão. Uma câmera já cadastrada também não pode trocar de sala mantendo o mesmo ID.

No host, o banco é `data/api/rag_audit.db`; dentro do container da API, o mesmo arquivo aparece como `/app/data/api/rag_audit.db`. Ferramentas executadas no host devem usar o primeiro caminho.

Se o servidor não tiver o ambiente Python instalado, a mesma simulação pode usar a imagem da API com um bind mount temporário do manifesto:

```powershell
docker compose --env-file .env.vision run --rm --no-deps `
  --volume "$($PWD.Path)\data\private\config\access-context.json:/run/secrets/access-context.json:ro" `
  --entrypoint python rag-audit `
  scripts/provision_access_context.py `
  /run/secrets/access-context.json `
  --database /app/data/api/rag_audit.db `
  --dry-run `
  --replace-assignments
```

Repita sem `--dry-run` após a revisão. O manifesto é visível somente nesse container descartável.

## Referências aprendidas

A aprendizagem fica desativada por padrão. Quando for habilitada após a calibração, uma observação elegível vira apenas um candidato `PENDING`. Ela não entra imediatamente na busca. Um responsável precisa revisar e executar uma das ações:

- `APPROVED`: passa a poder ser carregada pelo mesmo modelo e fingerprint;
- `REJECTED`: candidato recusado;
- `REVOKED`: referência anteriormente aprovada deixa de ser usada.

O vínculo com a versão e o fingerprint do modelo evita reaproveitar embeddings após uma troca incompatível. O worker recarrega o conjunto aprovado no intervalo configurado e aplica aprovação ou revogação sem reiniciar; se o banco aprendido falhar, mantém somente a galeria oficial. Também devem ser verificados identidade ativa, proveniência, qualidade, similaridade, margem e limite de referências por pessoa. A revisão é feita por `scripts/manage_learned_gallery.py`.

```powershell
python scripts\manage_learned_gallery.py `
  --database data\private\learned\learned.db `
  list --status PENDING
python scripts\manage_learned_gallery.py `
  --database data\private\learned\learned.db `
  show <id>
python scripts\manage_learned_gallery.py `
  --database data\private\learned\learned.db `
  approve <id> `
  --operator "usuario.corporativo" `
  --evidence-dir data\private\evidence
python scripts\manage_learned_gallery.py `
  --database data\private\learned\learned.db `
  reject <id> `
  --operator "usuario.corporativo" --reason "identidade divergente"
```

## Evidências privadas

Quando habilitada, a evidência é armazenada fora do banco principal como cena JPEG e miniatura. O evento contém somente uma referência aleatória de 64 caracteres hexadecimais.

O armazenamento privado aplica:

- TTL configurável;
- cota total e limite por item;
- SHA-256 de cena e miniatura;
- verificação de integridade durante a leitura;
- expurgo de itens vencidos;
- nomes não derivados de pessoa, câmera ou horário;
- permissões restritas no sistema de arquivos.

Fotos não entram no webhook, no PDF ou na outbox externa. O acesso pelo painel continua administrativo. O volume deve ser criptografado e a retenção precisa ser aprovada pelo controlador dos dados.

API e worker devem usar exatamente os mesmos valores de TTL, cota, limite por item e política de descarte. O índice privado grava essa política e recusa uma configuração divergente, evitando que dois processos apliquem retenções diferentes ao mesmo conjunto de fotos.

## Operação offline

Depois que uma entrada é confirmada, o evento vai para uma outbox SQLite antes da chamada HTTP. Se a API estiver indisponível, o item permanece local e é reenviado com idempotência. A fila também precisa continuar sendo drenada durante falhas ou reconexões do RTSP; indisponibilidade da câmera não deve impedir o reenvio de um evento que já foi persistido.

Produção e dry-run usam bancos separados, ambos contendo `{camera_id}` no caminho. Falhas transitórias ficam como `RETRYING`; respostas HTTP 4xx permanentes ficam como `DEAD` para inspeção do operador. A API aceita eventos autenticados vindos da fila somente até `RAG_AUDIT_QUEUED_EVENT_MAX_AGE_SECONDS`. Essa janela não é retenção ilimitada.

No dry-run, evidências ficam desativadas por padrão (`RAG_AUDIT_VISION_DRY_RUN_SAVE_EVIDENCE=false`) e a fila simulada tem retenção e quantidade máximas próprias. Esses limites evitam que um ensaio prolongado ocupe o disco reservado à operação.

```powershell
python scripts\manage_vision_outbox.py `
  --database data\private\outbox\cam-ti-01.db `
  summary
python scripts\manage_vision_outbox.py `
  --database data\private\outbox\cam-ti-01.db `
  list --status RETRYING --status DEAD
```

A ferramenta não mostra payloads. Reenfileirar um item `DEAD` ou excluir um item terminal exige `--operator`, `--reason` e `--confirm <event_id>` e grava uma ação administrativa no próprio banco.

## Executar a API no Windows

Requer Python 3.11.

```powershell
cd C:\caminho\para\rag-audit
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.api.example .env.api
uvicorn app.main:app --reload
```

Preencha `.env.api` antes de iniciar. O exemplo é um molde de produção e contém valores que precisam ser substituídos. Para usar somente os cenários simulados no computador de desenvolvimento, altere temporariamente `RAG_AUDIT_ENV=development` e `RAG_AUDIT_SEED_DEMO_DATA=true`; nunca use essa combinação no piloto. Abra `http://127.0.0.1:8000/dashboard`.

Para enviar os cenários simulados:

```powershell
python scripts\send_demo_events.py
```

Resultado esperado:

```text
Lucas:   AUTHORIZED / LOW
Mariana: JUSTIFIED  / MEDIUM
Roberto: ANOMALY    / HIGH
```

## Preparar a visão

Instale as dependências separadas:

```powershell
python -m pip install -r requirements-vision.txt
Copy-Item .env.vision.example .env.vision
```

Em seguida:

```powershell
python scripts\run_vision_worker.py --check-offline
python scripts\probe_intelbras_camera.py
python scripts\run_vision_worker.py --check
python scripts\run_vision_worker.py
```

O primeiro ciclo deve permanecer com `RAG_AUDIT_VISION_DRY_RUN=true`. O envio só pode ser habilitado depois da avaliação descrita nos documentos de implantação.

## Docker

```powershell
Copy-Item .env.api.example .env.api
Copy-Item .env.vision.example .env.vision
New-Item -ItemType Directory -Force `
  data\api, `
  data\private\cache, `
  data\private\config, `
  data\private\gallery, `
  data\private\learned, `
  data\private\outbox, `
  data\private\evidence, `
  data\vision-models | Out-Null
Copy-Item config\camera-calibration.example.json `
  data\private\config\camera-calibration.json

docker compose --env-file .env.vision --profile vision build
# Aplique o manifesto de contexto pelo fluxo descrito acima.
docker compose --env-file .env.vision up -d rag-audit
docker compose --env-file .env.vision --profile vision up -d vision-worker
```

Em produção, a API só fica pronta quando o conjunto de políticas configurado está no banco. Provisione o contexto antes de esperar o healthcheck verde. O worker só deve subir depois que galeria, calibração, bundle e fingerprint estiverem presentes.

Os dois containers executam como UID/GID `10001`, usam raiz somente leitura, removem capabilities Linux e habilitam `no-new-privileges`. O worker também:

- usa `/tmp` temporário com restrições;
- monta o bundle de modelos como somente leitura.

O Compose separa banco da API, galeria oficial somente leitura, cache de embeddings, calibração somente leitura, referências aprendidas, outbox e evidências. A API não recebe a galeria biométrica, a senha RTSP ou os modelos. O worker não recebe a senha administrativa nem o webhook. `--env-file .env.vision` é necessário para que o Compose resolva `RAG_AUDIT_VISION_MODELS_HOST_DIR` no bind mount; o conteúdo dos serviços continua vindo de `.env.api` e `.env.vision`.

No Linux, prepare os bind mounts antes de subir:

```bash
install -d -m 0750 \
  data/api \
  data/private/cache \
  data/private/config \
  data/private/gallery \
  data/private/learned \
  data/private/outbox \
  data/private/evidence \
  data/vision-models
cp config/camera-calibration.example.json \
  data/private/config/camera-calibration.json
sudo chown -R 10001:10001 data/api data/private data/vision-models
sudo find data/api data/private data/vision-models -type d -exec chmod 0750 {} \;
sudo find data/api data/private data/vision-models -type f -exec chmod 0640 {} \;
```

A galeria oficial permanece somente leitura. O cache derivado é gravado em `data/private/cache/<camera_id>/embeddings.arcface.npz`, pode ser apagado e será reconstruído a partir da galeria verificada.

No Docker Desktop para Windows, crie os mesmos diretórios com PowerShell, mantenha-os restritos à conta do serviço e à equipe administradora e valide escrita com `docker compose ... run --rm`. O mapeamento UID/GID do bind mount é mediado pela VM do Docker Desktop; não execute `chown` do Linux no NTFS.

## Contrato do evento

Endpoint: `POST /v1/webhooks/access-events`

Cabeçalho: `X-Camera-Key`, com a chave correspondente ao `camera_id` no mapa `RAG_AUDIT_CAMERA_API_KEYS_JSON`.

```json
{
  "event_id": "entry:vis-...",
  "camera_id": "cam-ti-01",
  "user_id": "EMP001",
  "room_id": "sala_ti_01",
  "timestamp": "2026-07-14T14:00:00-03:00",
  "door_result": "NOT_REPORTED",
  "recognition_confidence": 0.78,
  "identity_status": "MATCHED",
  "entry_evidence": "VISION_LINE_CROSSING",
  "recognition_source": "LOCAL_ARCFACE",
  "recognition_model": "insightface-buffalo_l-scrfd10g-w600k-r50",
  "recognition_model_fingerprint": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
}
```

Regras principais:

- `event_id` é idempotente;
- `timestamp` precisa incluir offset;
- `room_id` deve corresponder à câmera cadastrada;
- eventos somente visuais usam `door_result=NOT_REPORTED`;
- `LOCAL_ARCFACE` exige o nome do motor e o SHA-256 do bundle verificado;
- `LOCAL_ARCFACE` já foi aceito pelo worker com os limiares calibrados; a API não reaplica o `recognition_threshold` legado da câmera;
- `recognition_threshold` continua válido para fontes externas, como integrações Intelbras;
- confiança não concede acesso e não substitui revisão;
- imagem, embedding, senha e base64 são proibidos no payload.

## Avaliação e calibração

O limiar não deve ser escolhido por impressão visual nem copiado de benchmark. A avaliação no ambiente real deve separar identidades usadas para ajuste das usadas para validação e registrar:

- `FPIR`: proporção de buscas que atribuem uma identidade incorreta;
- `FNIR`: proporção de buscas de pessoas cadastradas que não são identificadas;
- rank-1: frequência em que a identidade correta é o primeiro candidato;
- taxa de `UNKNOWN`, `AMBIGUOUS` e `LOW_QUALITY`;
- duplicidade de entradas;
- erro de direção;
- latência por etapa e ponta a ponta.

O conjunto precisa incluir entrada, saída, parada na porta, duas pessoas, desconhecidos, óculos, boné, variação de altura, luz baixa, contraluz e movimento. `scripts/evaluate_vision_dataset.py` varre qualidade, similaridade e margem e só recomenda pontos cujo limite superior de confiança do FPIR condicional respeite o teto informado. O relatório também mantém o FPIR operacional, incluindo falhas de aquisição no denominador. Resultados e imagens reais devem permanecer fora do Git.

A ferramenta é frame-level: aceita uma ROI normalizada opcional, exige uma face por probe e não executa tracking, consenso ou travessia. Ela registra no relatório providers, tamanho do detector, política de qualidade e ROI usados. Isso auxilia a escolha inicial dos limiares, mas não substitui a validação temporal do pipeline filmado na porta.

O objetivo de calibração é escolher o menor FNIR possível sem ultrapassar o FPIR aceito pelo responsável. Se não houver amostra suficiente para estimar falso positivo, o sistema deve continuar em dry-run.

## Testes

```powershell
pytest
python -m ruff check app scripts tests
```

Os testes usam arquivos temporários e não alteram `data/rag_audit.db`.

## Estrutura

```text
app/
  main.py
  risk_engine.py
  reports.py
  vision/
    recognizer.py
    face_tracking.py
    entry_tracker.py
    learned.py
    evidence.py
    outbox.py
    pipeline.py
    worker.py
scripts/
  evaluate_vision_dataset.py
  import_incontrol_gallery.py
  manage_learned_gallery.py
  manage_vision_outbox.py
  provision_access_context.py
  prepare_vision_models.py
  verify_vision_models.py
  probe_intelbras_camera.py
  run_vision_worker.py
docs/
  integracao-vipc-1230-g2.md
  plano-implantacao-ambiente-intelbras.md
tests/
```

## Limites e segurança

A VIPC 1230 G2 é uma câmera RGB comum. O projeto não possui mecanismo de liveness validado para esse cenário; uma fotografia, tela ou máscara pode enganar o reconhecimento. O sistema também não recebe a senha digitada, o estado da fechadura nem um contato de porta. Logo:

- não usar o resultado para abrir ou bloquear a porta;
- não afirmar que uma senha foi aceita;
- não aplicar punição ou decisão trabalhista automaticamente;
- encaminhar `UNKNOWN`, `AMBIGUOUS` e divergências para revisão humana;
- manter canal de contestação e correção;
- restringir acesso à biometria e às evidências;
- documentar finalidade, hipótese legal, necessidade, retenção e descarte.

Biometria é dado pessoal sensível na LGPD. O projeto é uma implementação técnica, não um parecer jurídico nem uma certificação. Antes de um piloto real, envolva o encarregado/DPO e o jurídico, registre as decisões e faça uma avaliação de impacto.

O roteiro específico está em [Integração da Intelbras VIPC 1230 G2](docs/integracao-vipc-1230-g2.md). A preparação da visita, testes e critérios de aceite estão no [Plano de implantação no ambiente Intelbras](docs/plano-implantacao-ambiente-intelbras.md).
