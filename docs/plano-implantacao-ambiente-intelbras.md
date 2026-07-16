# Plano de implantação no ambiente Intelbras

## 1. Objetivo e critério de linguagem

O piloto conectará uma Intelbras VIPC 1230 B/D G2 fixa, uma galeria facial autorizada e o RAG-Audit para registrar pessoas observadas entrando na sala.

A porta abre por senha, mas a aplicação não recebe:

- a senha digitada;
- o resultado do teclado;
- o estado da fechadura;
- a abertura física da porta.

O resultado correto é “entrada observada pela câmera”. `door_result` permanece `NOT_REPORTED`. Nenhum relatório deve afirmar que a senha foi aceita ou que a fechadura concedeu o acesso.

O sistema também não tem liveness validado. Ele é uma ferramenta de auditoria com revisão humana, não uma fonte única para punição, bloqueio ou decisão trabalhista.

## 2. Resultado esperado da visita

Ao final da primeira sessão técnica, o objetivo não é colocar alertas em produção. O objetivo é sair com dados suficientes para responder:

1. O rosto aparece com tamanho e qualidade utilizáveis?
2. A trajetória de entrada pode ser separada de saída e permanência?
3. As fotos da base Intelbras podem ser exportadas por um caminho autorizado?
4. Qual é a taxa inicial de falso positivo e falso negativo?
5. O servidor consegue manter RTSP, inferência e outbox com estabilidade?
6. A retenção, o acesso e a revisão foram aprovados?

Se alguma resposta permanecer aberta, o worker continua em dry-run.

## 3. Arquitetura de implantação

```text
                    +--------------------------+
                    | Base facial autorizada   |
                    | InControl / Defense IA / |
                    | controlador específico   |
                    +-------------+------------+
                                  |
                              IDs + fotos
                                  |
                                  v
+------------------+   RTSP   +---+-------------------------+
| VIPC 1230 G2     +--------->| worker local de visão       |
| fixa na sala     |          | ROI, SCRFD, ArcFace,        |
+--------+---------+          | tracking, consenso, entrada |
         |                    +---+-------------------------+
         | SMTP/foto opcional     |
         +------------------------+ gatilho complementar
                                  |
                       evento em outbox, sem imagem
                                  |
                                  v
                    +-------------+-------------+
                    | API RAG-Audit             |
                    | contexto, log, alerta,    |
                    | dashboard e PDF           |
                    +---------------------------+
```

O modo padrão é `entry`, com sequência porta, linha no sentido `IN` e zona interna. O modo `door` é somente observação e deve ficar em dry-run, salvo override operacional explícito.

## 4. O que existe no repositório

Os componentes disponíveis cobrem:

- teste seguro de conectividade e leitura RTSP;
- captura de frame para calibração;
- importação defensiva de ZIP e diretório de fotos;
- manifesto de galeria com hashes;
- InsightFace `buffalo_l` com SCRFD-10GF e ArcFace W600K R50;
- validação de embeddings, arquivos da galeria e qualidade da face;
- tracking com sinais geométricos e de aparência;
- consenso temporal por trilha;
- rastreamento direcional com deadband, segmento e deslocamento mínimo;
- estados `MATCHED`, `UNKNOWN`, `AMBIGUOUS` e `LOW_QUALITY`;
- outbox SQLite idempotente;
- candidatos de aprendizagem `PENDING`, com aprovação, rejeição e revogação;
- armazenamento de evidência com referência opaca, TTL, cota e integridade;
- bundle externo de modelos com manifesto e SHA-256;
- container de visão sem root, com raiz e modelos somente leitura;
- testes e CI.

A integração do worker deve manter estes vínculos:

- evento e melhor evidência indexados por trilha e identidade;
- troca de identidade reinicia consenso, trajetória e amostras;
- referências aprendidas só entram após aprovação manual;
- queda de câmera não interrompe a drenagem da outbox;
- estados incompletos são encerrados sem fabricar entrada;
- `recognition_source` é `LOCAL_ARCFACE`.

O receptor de e-mail não deve ser considerado concluído antes da análise de um `.eml` original da câmera.

## 5. Informações a levantar antes da visita

Não enviar credenciais por chat, chamado, e-mail ou commit. Elas serão digitadas no ambiente durante a sessão.

### 5.1 Câmera

- modelo exato: B G2 ou D G2;
- número de série e MAC, mantidos no inventário interno;
- firmware e versão da interface web;
- IP, máscara, gateway e VLAN;
- porta RTSP configurada;
- codec, resolução, bitrate e FPS de cada stream;
- NTP, fuso e horário exibido;
- posição física e altura;
- distância da câmera até a soleira;
- largura da passagem;
- existência de vidro, espelho, janela ou monitor no campo;
- condição de luz durante o dia e a noite;
- possibilidade de ajustar inclinação sem trocar a câmera.

### 5.2 Fechadura

- modelo do teclado ou controle;
- confirmação documentada de que não há log/API disponível;
- tempo médio entre digitação e abertura;
- sentido e largura da porta;
- comportamento quando duas pessoas passam na mesma abertura;
- existência de contato magnético ou sensor, mesmo que ainda não integrado.

### 5.3 Galeria Intelbras

Identificar a origem real:

- InControl Web;
- Defense IA;
- controlador facial independente;
- outro software do integrador.

Registrar versão, forma oficial de exportação e identificador estável da pessoa. Para o piloto, preparar apenas 5 a 15 colaboradores informados e autorizados, com pelo menos uma foto válida por pessoa. Não copiar a base inteira.

Templates biométricos proprietários não servem como fotos nem como embeddings ArcFace. Se o sistema não exportar imagens por um método suportado, a solução precisa ser discutida com Intelbras ou integrador.

### 5.4 Servidor

- sistema operacional e versão;
- CPU, RAM e espaço livre;
- Docker disponível ou execução Python;
- criptografia de disco;
- conta de serviço;
- política de backup;
- antivírus/EDR;
- NTP;
- IP e VLAN;
- janela de manutenção;
- monitoramento e destino de logs;
- política para impedir suspensão ou hibernação.

### 5.5 Governança

- responsável técnico;
- proprietário da sala;
- encarregado/DPO e jurídico;
- finalidade e hipótese legal;
- grupo inicial do piloto;
- quem revisa desconhecidos e ambiguidades;
- limite de FPIR aceito;
- prazo de retenção de cena, miniatura, evento e relatório;
- processo de contestação;
- resposta a incidente;
- canal de alerta aprovado.

## 6. Rede e contas

Fluxos mínimos:

| Origem | Destino | Porta | Uso |
|---|---|---:|---|
| worker | câmera | TCP 554 | RTSP |
| câmera e servidor | NTP interno | UDP 123 | sincronização |
| estação administrativa | câmera | TCP 443 | configuração |
| estação administrativa | API | TCP 443 | dashboard, via proxy |
| worker | API | TCP 8000 ou porta interna | webhook |
| API | automação corporativa | TCP 443 | alertas |
| câmera | relay SMTP | porta aprovada | foto opcional |
| receptor | caixa corporativa | TCP 993 | IMAP opcional |

Regras:

- câmera em VLAN de CFTV/IoT;
- ACL somente para os fluxos necessários;
- nenhum RTSP, SMTP ou painel exposto à internet;
- usuário RTSP exclusivo, somente leitura;
- senha única e armazenada em segredo;
- administração da câmera somente por HTTPS quando suportado;
- NTP comum entre câmera, worker e API;
- SMTP local nunca configurado como relay aberto;
- API atrás de TLS, identidade corporativa, RBAC e MFA antes de produção.

## 7. Preparar o repositório no servidor

### 7.1 Python

```powershell
cd C:\caminho\para\rag-audit
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements-vision.txt
Copy-Item .env.api.example .env.api
Copy-Item .env.vision.example .env.vision
```

Preencher os arquivos apenas no servidor:

- `.env.api`: senha administrativa, chaves por câmera, webhook, política e banco;
- `.env.vision`: RTSP, chave da câmera correspondente, modelos e parâmetros faciais.

Em produção, `RAG_AUDIT_CAMERA_API_KEYS_JSON` é obrigatório na API e mapeia cada `camera_id` para sua própria chave. O worker recebe somente `RAG_AUDIT_CAMERA_API_KEY`, com o valor correspondente à câmera que ele processa. No primeiro ciclo:

```text
RAG_AUDIT_VISION_MODE=entry
RAG_AUDIT_VISION_DRY_RUN=true
RAG_AUDIT_LEARN_ENABLED=false
```

Os nomes finais e todos os limites devem seguir os dois exemplos versionados. Não guardar uma cópia preenchida fora do cofre ou do host. Variáveis já exportadas no processo têm precedência; `.env` permanece somente como fallback legado durante a migração.

### 7.2 Docker

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

docker compose --env-file .env.vision build
docker compose --env-file .env.vision --profile vision build
```

O `--env-file .env.vision` não mistura os segredos dos serviços. Ele permite ao Compose resolver `RAG_AUDIT_VISION_MODELS_HOST_DIR` antes de criar o bind mount; a API continua lendo `.env.api` e o worker, `.env.vision`.

API e worker executam como `10001:10001`, com raiz somente leitura, capabilities removidas e `no-new-privileges`. Os mounts são:

| Host | Container | Serviço | Modo |
|---|---|---|---|
| `data/api` | `/app/data/api` | API | leitura/escrita |
| `data/private/evidence` | `/app/data/private/evidence` | API e worker | leitura/escrita |
| `data/private/gallery` | `/app/data/private/gallery` | worker | somente leitura |
| `data/private/cache` | `/app/data/private/cache` | worker | leitura/escrita |
| `data/private/config` | `/app/data/private/config` | worker | somente leitura |
| `data/private/learned` | `/app/data/private/learned` | worker | leitura/escrita |
| `data/private/outbox` | `/app/data/private/outbox` | worker | leitura/escrita |
| `data/vision-models` | `/app/vision-models` | worker | somente leitura |

A API não recebe galeria, calibração, outbox, modelos ou senha RTSP. O worker não recebe senha administrativa nem webhook.

A galeria oficial permanece somente leitura. `RAG_AUDIT_GALLERY_CACHE_PATH` aponta para `data/private/cache/{camera_id}/embeddings.arcface.npz`, em volume gravável separado. Esse arquivo é derivado, não substitui as fotos e pode ser apagado para reconstrução após uma mudança validada.

### 7.3 Permissões no host Linux

Execute antes do primeiro `docker compose up`:

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

Depois do `chown`, prefira executar o provisionamento pelo container descartável mostrado na seção 10.1. Executar a ferramenta no host com outro usuário pode criar `rag_audit.db` com proprietário incompatível com o UID `10001`.

Depois que os arquivos oficiais forem copiados, mantenha galeria, calibração e modelos sem escrita para processos comuns. Para confirmar os mounts:

```bash
docker compose --env-file .env.vision run --rm --entrypoint sh rag-audit \
  -ec 'test -w /app/data/api && test -w /app/data/private/evidence'
docker compose --env-file .env.vision --profile vision run --rm \
  --entrypoint sh vision-worker \
  -ec 'test -r /app/data/private/gallery/manifest.json &&
       test -w /app/data/private/cache &&
       test -r /app/data/private/config/camera-calibration.json &&
       test -w /app/data/private/outbox &&
       test -w /app/data/private/learned &&
       test -w /app/data/private/evidence &&
       test -r /app/vision-models/bundle-manifest.json'
```

### 7.4 Diretórios no Windows

No Docker Desktop, o NTFS é compartilhado pela VM e `chown 10001:10001` não deve ser executado no host. Crie os diretórios com PowerShell, limite as ACLs à conta de serviço e ao grupo administrador e use os mesmos testes de mount:

```powershell
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

docker compose --env-file .env.vision run --rm `
  --entrypoint sh rag-audit `
  -ec "test -w /app/data/api && test -w /app/data/private/evidence"
docker compose --env-file .env.vision --profile vision run --rm `
  --entrypoint sh vision-worker `
  -ec "test -w /app/data/private/cache && test -w /app/data/private/outbox && test -w /app/data/private/learned && test -w /app/data/private/evidence"
```

Use BitLocker ou outro mecanismo corporativo para criptografar o volume. Não compartilhe `data/private` por SMB com usuários comuns.

## 8. Preparar e verificar os modelos

O aplicativo não baixa pesos. O responsável deve obter o `buffalo_l` por uma fonte autorizada e confirmar a licença do uso pretendido.

Arquivos obrigatórios:

```text
data/vision-models/models/buffalo_l/det_10g.onnx
data/vision-models/models/buffalo_l/w600k_r50.onnx
```

Gerar o manifesto:

```powershell
python scripts\prepare_vision_models.py `
  --root data\vision-models `
  --license-reference "aprovação ou registro interno" `
  --accept-model-license
```

Copiar o fingerprint exibido para:

```text
RAG_AUDIT_VISION_MODEL_BUNDLE_SHA256=<64 caracteres hexadecimais>
```

Verificar:

```powershell
python scripts\verify_vision_models.py `
  --root data\vision-models `
  --expected-fingerprint $env:RAG_AUDIT_VISION_MODEL_BUNDLE_SHA256
```

Não corrigir uma falha de hash atualizando o valor às cegas. Primeiro confirmar por que o arquivo mudou.

## 9. Testar a câmera

### 9.1 Conferir configuração

Começar pelo stream principal:

- 1920×1080;
- H.264;
- FPS estável;
- bitrate suficiente para não destruir detalhes do rosto;
- exposição que preserve o rosto;
- DWDR/BLC avaliados com a porta aberta;
- sem áudio;
- NTP correto.

Evitar alterar vários parâmetros ao mesmo tempo. Registrar valor anterior, valor novo e efeito observado.

### 9.2 Provar conectividade

```powershell
python scripts\probe_intelbras_camera.py
```

Resultado esperado:

- TCP 554 acessível;
- autenticação aceita;
- um frame lido e descartado;
- nenhuma credencial impressa.

Uma falha aqui deve ser resolvida na rede ou câmera antes de ajustar reconhecimento.

## 10. Importar a galeria de teste

Exemplo:

```powershell
python scripts\import_incontrol_gallery.py `
  C:\Temp\incontrol-piloto.zip `
  data\private\gallery

python scripts\sync_gallery_people.py `
  --database data\api\rag_audit.db `
  data\private\gallery\manifest.json
```

Conferir:

- quantidade de pessoas;
- ID externo estável;
- associação entre ID e nome;
- hash e legibilidade das fotos;
- ausência de duplicatas;
- ausência de senha, cartão e campos desnecessários;
- autorização de cada pessoa no piloto.

Sincronizar uma pessoa não concede permissão de sala nem cria escala. Esses dados continuam sob a regra de negócio da API.

### 10.1 Provisionar o contexto da API

Copie o modelo para um caminho privado:

```powershell
Copy-Item config\access-context.example.json `
  data\private\config\access-context.json
```

Preencha salas, câmeras, pessoas, permissões, escalas e políticas. O arquivo:

- não contém senha, token, chave da API, host da câmera ou URL RTSP;
- usa IDs estáveis e iguais aos IDs sincronizados da galeria;
- inclui fusos IANA;
- inclui as três decisões `AUTHORIZED`, `JUSTIFIED` e `ANOMALY` na versão configurada;
- deve ser revisado por quem responde pela autorização física da sala.

Simule primeiro:

```powershell
python scripts\provision_access_context.py `
  data\private\config\access-context.json `
  --database data\api\rag_audit.db `
  --dry-run `
  --replace-assignments
```

O dry-run usa uma transação revertida e não cria o banco quando ele ainda não existe. Revise quantos registros seriam criados, atualizados, mantidos ou removidos. Faça backup do banco e aplique:

```powershell
python scripts\provision_access_context.py `
  data\private\config\access-context.json `
  --database data\api\rag_audit.db `
  --replace-assignments
```

`--replace-assignments` remove permissões e escalas ausentes somente para pessoas listadas no manifesto. Sem a opção, o comando faz merge e uma autorização antiga omitida continuaria no banco. Pessoas fora do manifesto não são removidas automaticamente; para desligá-las, inclua-as com `active=false` e trate a revogação das atribuições de forma explícita.

Políticas são imutáveis por `policy_id/version`. Uma mudança de texto, decisão ou códigos exige nova versão e atualização coordenada de `RAG_AUDIT_POLICY_VERSION`. Uma câmera não pode trocar de sala preservando o mesmo `camera_id`; um remanejamento físico usa novo ID.

Ferramentas no host usam `data/api/rag_audit.db`. `/app/data/api/rag_audit.db` é o caminho do mesmo bind mount dentro do container e não deve ser passado a um comando executado no Windows ou Linux host.

Quando o host não tiver Python, use a imagem da API e monte o manifesto somente no container descartável:

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

Repita sem `--dry-run` somente após revisar o resumo. O serviço normal da API não recebe esse mount.

O `recognition_threshold` da câmera continua sendo usado por fontes externas. Eventos `LOCAL_ARCFACE` já foram aceitos no worker pelos limiares calibrados e pelo consenso temporal, portanto a API não reaplica esse valor legado e exige o nome do motor mais o fingerprint SHA-256 verificado.

## 11. Capturar a referência do enquadramento

Com o local vazio e autorização para a captura:

```powershell
python scripts\capture_calibration_frame.py --empty-room-confirmed
```

Criar o arquivo privado:

```text
data/private/config/camera-calibration.json
```

a partir de:

```text
config/camera-calibration.example.json
```

Definir:

- `door_zone`;
- `inside_zone`;
- início e fim de `entry_line`;
- lado interno;
- observações mínimas em cada zona;
- timeout e cooldown;
- `line_deadband`;
- margem do segmento;
- deslocamento mínimo;
- tempo máximo de transição.

A ROI do detector deve envolver essas regiões com uma margem curta. O frame de calibração não deve permanecer além do necessário.

O check precisa recusar polígonos auto-intersectantes, linha invertida e segmento que não atravesse o caminho entre os centros das zonas. Ainda assim, validar entrada e saída reais depois de qualquer mudança física na câmera.

## 12. Diagnóstico da imagem

Antes de medir identidade, coletar amostras controladas de cada participante:

- entrada olhando naturalmente;
- entrada rápida;
- saída;
- parada e recuo;
- centro e bordas da porta;
- óculos;
- boné, quando fizer parte do uso normal;
- luz diurna;
- luz noturna;
- porta aberta com contraluz.

Para cada captura, registrar:

- IED;
- tamanho do rosto;
- nitidez;
- brilho e contraste;
- yaw, pitch e roll;
- detecção válida ou rejeitada;
- posição na imagem;
- stream usado.

Se o IED for baixo em quase toda a soleira:

1. reposicionar a câmera dentro das possibilidades;
2. estreitar a ROI;
3. usar o stream principal;
4. elevar bitrate e conferir foco;
5. melhorar a luz frontal;
6. reduzir a velocidade de obturador somente se isso não escurecer o rosto;
7. repetir a medição.

Diminuir o limiar de similaridade não recupera detalhe que a câmera não capturou.

## 13. Verificação estática do worker

Antes da visita, sem abrir o RTSP:

```powershell
python scripts\run_vision_worker.py --check-offline
```

Na rede da câmera:

```powershell
python scripts\run_vision_worker.py --check
```

O check deve validar:

- configuração;
- bundle e fingerprint;
- providers ONNX disponíveis;
- galeria e hashes;
- geometria;
- acesso de escrita a outbox e evidências;
- formato da URL da API sem expor segredos.

O modo completo acrescenta alcance TCP e leitura de um frame da câmera. A autenticação da API só é exercitada quando um evento real ou controlado é enviado; o check não fabrica uma entrada.

Falha de qualquer dependência obrigatória impede a ativação.

## 14. Dry-run direcional

Manter:

```text
RAG_AUDIT_VISION_MODE=entry
RAG_AUDIT_VISION_DRY_RUN=true
RAG_AUDIT_VISION_DRY_RUN_SAVE_EVIDENCE=false
RAG_AUDIT_LEARN_ENABLED=false
```

A outbox simulada aplica `RAG_AUDIT_VISION_DRY_RUN_OUTBOX_RETENTION_DAYS` e `RAG_AUDIT_VISION_DRY_RUN_OUTBOX_MAX_EVENTS`. Ajuste esses limites ao período do ensaio, sem reutilizar a fila de produção.

Executar:

```powershell
python scripts\run_vision_worker.py
```

Observar pelo menos:

- entrada de uma pessoa;
- permanência no enquadramento;
- saída;
- recuo antes da linha;
- duas pessoas juntas;
- uma pessoa passando atrás de outra;
- desconhecido;
- pessoa cadastrada com captura ruim;
- perda e retorno de RTSP;
- API desligada e religada.

O modo `door` pode ser usado temporariamente para entender a zona e a qualidade, mas não entra nos indicadores de entrada sem override explícito. Não ajustar `door` para parecer direcional.

## 15. Avaliar identidade

A avaliação precisa de dois grupos de imagens:

- calibração: usada para escolher qualidade, limiar e margem;
- validação: pessoas e tentativas não usadas no ajuste.

O manifesto do conjunto deve apontar para arquivos privados, sem copiar fotos para o Git. Formato mínimo:

```json
{
  "schema_version": 1,
  "gallery_manifest": "gallery/manifest.json",
  "gallery_manifest_sha256": "<sha256>",
  "probes": [
    {
      "kind": "known",
      "external_id": "EMP001",
      "image_path": "probes/known-001.jpg",
      "image_sha256": "<sha256>"
    },
    {
      "kind": "unknown",
      "image_path": "probes/unknown-001.jpg",
      "image_sha256": "<sha256>"
    }
  ]
}
```

Todos os caminhos são relativos ao manifesto e todos os hashes são obrigatórios. Executar:

```powershell
python scripts\evaluate_vision_dataset.py <manifesto-privado> `
  --max-fpir 0.01 `
  --json-output data\private\evaluation\result.json `
  --csv-output data\private\evaluation\grid.csv
```

Ela produz, por limiar de qualidade, similaridade e margem:

- FPIR;
- FNIR;
- rank-1;
- taxa de identificação verdadeira;
- taxa de identificação incorreta;
- taxa de ambiguidade;
- taxa de baixa qualidade;
- taxa de falha de detecção;
- contagem de probes sem face ou com múltiplas faces.

Essa ferramenta mede reconhecimento **frame a frame** e exige exatamente uma face por probe. Use `--roi left,top,right,bottom` quando quiser reproduzir o recorte normalizado da implantação; sem essa opção, usa a imagem inteira. Ela não reproduz tracking, consenso, deduplicação nem travessia da porta. O relatório registra provider, `det-size`, qualidade de galeria, política facial e ROI. O aceite de produção ainda exige a avaliação temporal da seção seguinte, com a configuração efetiva do worker.

O relatório produz dois valores: FPIR operacional, com todos os probes desconhecidos, e FPIR condicional, somente entre probes que chegaram à comparação. Falhas `NO_FACE`, múltiplas faces ou baixa qualidade podem deixar o operacional artificialmente otimista; a recomendação automática usa o limite superior de Wilson do FPIR condicional.

Definições usadas no aceite:

- **FPIR:** proporção de tentativas em que o sistema atribui uma identidade errada;
- **FNIR:** proporção de tentativas de pessoa cadastrada em que a identidade não é aceita;
- **rank-1:** proporção em que a pessoa correta aparece no topo, independentemente da aceitação;
- **failure to acquire:** rosto que não chega à comparação por qualidade ou detecção.

O limiar deve ser escolhido pelo limite de falso positivo aceito, não pelo maior percentual geral. Em auditoria de acesso, dar o nome errado costuma ser mais grave do que deixar um caso como desconhecido.

## 16. Avaliar a entrada

Identidade correta não basta. Medir separadamente:

- direção correta;
- entradas perdidas;
- saídas contadas como entrada;
- duplicidade por fragmentação de track;
- duas pessoas na mesma abertura;
- pessoa parada na linha;
- cruzamento fora do segmento;
- jitter na zona morta;
- latência entre porta, zona interna e evento;
- recontagem após sair e voltar.

Cada teste deve ter um gabarito produzido por observação humana, com horário e cenário. Não usar o próprio resultado do sistema como verdade de referência.

## 17. Consenso temporal

Começar com configuração conservadora:

- histórico suficiente para absorver frames ruins;
- múltiplos matches;
- proporção mínima alta;
- sequência consecutiva;
- margem top-1/top-2;
- reinício quando a identidade muda.

Frames `UNKNOWN`, `AMBIGUOUS` e `LOW_QUALITY` precisam impedir que poucas observações antigas confirmem uma identidade. Baixa qualidade também quebra a sequência consecutiva; não deve preservar silenciosamente o consenso de uma pessoa quando a aparência da trilha mudou.

Se o sistema demora demais:

1. medir tempo por etapa;
2. conferir FPS processado;
3. reduzir ROI;
4. revisar qualidade da captura;
5. só então estudar quantidade de frames.

Não compensar latência aceitando um único frame.

## 18. Aprendizagem supervisionada

Durante toda a calibração:

```text
RAG_AUDIT_LEARN_ENABLED=false
```

Quando a equipe decidir testar o recurso, a ativação apenas permite criar candidatos `PENDING`. Nenhum candidato entra na galeria ativa sem revisão.

Fluxo:

```text
observação elegível
  -> PENDING
  -> revisão da pessoa, evidência, qualidade, similaridade e proveniência
     -> APPROVED
     -> REJECTED
     -> REVOKED, se uma aprovação deixar de ser válida
```

A ferramenta administrativa exige identificação do operador e motivo nas decisões aplicáveis:

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
python scripts\manage_learned_gallery.py `
  --database data\private\learned\learned.db `
  revoke <id> `
  --operator "usuario.corporativo" --reason "pessoa removida do piloto"
```

Regras:

- somente pessoas ativas;
- mesmo modelo e fingerprint;
- número limitado de referências;
- amostra com qualidade superior ao mínimo normal;
- consenso forte;
- similaridade e margem dentro da faixa aprovada;
- evidência disponível durante a revisão;
- operador e justificativa registrados.

Uma aprovação incorreta contamina comparações futuras. A revisão não deve ser tratada como tarefa automática.

O worker recarrega as referências aprovadas no intervalo configurado. Uma revogação deve desaparecer da busca sem reinício; se o banco de aprendizado falhar, somente a galeria oficial pode permanecer ativa.

## 19. Evidências

Configurar uma política aprovada para:

- TTL;
- cota total;
- tamanho máximo por item;
- expurgo;
- acesso administrativo;
- backup, se permitido;
- descarte em incidente ou revogação.

Validar:

1. referência opaca sem dados pessoais;
2. cena e miniatura com SHA-256;
3. leitura de arquivo íntegro;
4. rejeição de arquivo alterado;
5. desaparecimento após expiração;
6. comportamento ao atingir a cota;
7. permissão do diretório;
8. ausência de foto em webhook, PDF e alerta externo.

O volume deve estar criptografado. A referência opaca não torna a foto anônima; ela apenas evita expor identidade no nome do arquivo.

Repita os mesmos quatro valores de evidência em `.env.api` e `.env.vision`. O índice recusa políticas divergentes para impedir que API e worker apliquem TTL ou cotas diferentes. Em dry-run, mantenha `RAG_AUDIT_VISION_DRY_RUN_SAVE_EVIDENCE=false`, salvo autorização específica para revisar imagens do piloto.

## 20. Outbox e modo offline

Os caminhos precisam ser distintos e conter a câmera:

```text
RAG_AUDIT_VISION_OUTBOX_PATH=data/private/outbox/{camera_id}.db
RAG_AUDIT_VISION_DRY_RUN_OUTBOX_PATH=data/private/outbox/{camera_id}.dry-run.db
```

Mesmo no dry-run, o lease de exclusividade fica na outbox de produção para impedir dois workers concorrentes sobre a mesma câmera. Os eventos simulados continuam isolados no banco `.dry-run.db` e nunca devem ser copiados para produção.

Teste obrigatório do isolamento de dry-run:

1. iniciar o worker com dry-run;
2. produzir uma entrada controlada;
3. confirmar que o item foi persistido apenas no banco `.dry-run.db`;
4. confirmar que nenhum item foi enviado à API;
5. validar retenção e limite de quantidade;
6. nunca copiar esse backlog para produção.

Depois da aprovação formal para um ensaio de entrega real:

1. iniciar API e worker em modo de produção controlado;
2. desligar a API;
3. produzir uma entrada de teste autorizada;
4. confirmar persistência na outbox de produção;
5. interromper o RTSP;
6. religar a API mantendo a câmera indisponível;
7. confirmar que a fila continua sendo drenada;
8. reenviar o mesmo evento e confirmar idempotência;
9. religar a câmera e verificar reconexão sem entrada fantasma.

O worker deve registrar falhas sem imprimir URL RTSP, chave da API ou dados biométricos.

Estados operacionais:

- `PENDING`: persistido e ainda não enviado;
- `RETRYING`: falha transitória, com backoff;
- `SENT`: aceito ou reconhecido como duplicata idempotente;
- `DEAD`: resposta 4xx permanente ou item que exige correção humana.

A API usa a janela normal para eventos diretos e `RAG_AUDIT_QUEUED_EVENT_MAX_AGE_SECONDS` para eventos autenticados que ficaram na outbox. Um valor maior ajuda na indisponibilidade prolongada, mas precisa continuar limitado pela política de retenção e pela utilidade operacional do evento.

Consultar contagens e itens problemáticos sem imprimir o payload:

```powershell
python scripts\manage_vision_outbox.py `
  --database data\private\outbox\cam-ti-01.db `
  summary

python scripts\manage_vision_outbox.py `
  --database data\private\outbox\cam-ti-01.db `
  list --status RETRYING --status DEAD
```

Antes de agir sobre `DEAD`, confira o erro, o cadastro da câmera/sala/pessoa e a idade do evento. Não altere o payload diretamente para contornar a validação; corrija a origem e gere uma trilha operacional da decisão.

Depois de corrigir a causa:

```powershell
python scripts\manage_vision_outbox.py `
  --database data\private\outbox\cam-ti-01.db `
  requeue <event_id> `
  --confirm <event_id> `
  --operator "usuario.corporativo" `
  --reason "cadastro corrigido e validado"
```

Para excluir um item `SENT` após a retenção ou um `DEAD` que não pode ser reenviado, troque `requeue` por `delete` e informe um motivo específico. A ferramenta bloqueia exclusão de `PENDING` e `RETRYING` e registra a ação administrativa sem copiar o payload.

Sem Python no host, execute a ferramenta pela imagem do worker e use o caminho interno:

```powershell
docker compose --env-file .env.vision --profile vision run --rm --no-deps `
  --entrypoint python vision-worker `
  scripts/manage_vision_outbox.py `
  --database /app/data/private/outbox/cam-ti-01.db `
  summary
```

### 20.1 Migração de uma outbox legada

Se existir `vision-outbox.db` de uma versão anterior:

1. mantenha o worker parado;
2. faça uma cópia de segurança privada;
3. consulte status, `event_id`, câmera e idade dos itens;
4. resolva ou remova backlog inválido conforme a política;
5. confirme que todos os eventos pertencem a `cam-ti-01`;
6. somente então mova o arquivo para `data/private/outbox/cam-ti-01.db`;
7. inicialize o worker em dry-run e verifique contagens antes do envio real.

Não renomeie um banco desconhecido e não misture câmeras. Um `vision-outbox.db` usado em simulação deve ser arquivado ou descartado; nunca copie seu backlog para `cam-ti-01.db`.

## 21. E-mail com foto

O e-mail é uma segunda fase. Antes de escrever o receptor, coletar localmente:

- `.eml` original de teste;
- `.eml` original de movimento;
- dois alertas do mesmo movimento;
- anexo original;
- horários de captura, envio e chegada;
- firmware e configuração SMTP.

Não encaminhar o e-mail, pois o forward altera cabeçalhos e MIME. Guardar temporariamente em:

```text
data/private/samples/
```

O receptor deverá:

- usar caixa ou destinatário exclusivo;
- validar UID/UIDVALIDITY no IMAP;
- limitar mensagem, anexo, dimensões e pixels;
- validar JPEG/PNG/WebP pela assinatura;
- remover metadados desnecessários;
- bloquear conteúdo ativo e arquivos compactados;
- calcular hashes para idempotência;
- não confiar isoladamente em `From`, assunto ou `Message-ID`;
- persistir antes de marcar como processado;
- excluir temporários em sucesso e erro;
- correlacionar com a janela RTSP;
- gerar apenas observação quando não houver direção.

Como o e-mail costuma chegar depois da imagem, abrir o RTSP somente após o recebimento pode perder a entrada. O desenho preferido mantém RTSP e um buffer curto; o e-mail ajuda a localizar e correlacionar o evento.

## 22. Ativar envio real

Pré-condições:

- bundle e licença aprovados;
- câmera e NTP estáveis;
- ROI e trajetória validadas;
- FPIR dentro do limite formal;
- FNIR conhecido e aceito;
- rank-1 medido;
- desconhecidos e ambiguidades com revisão;
- outbox validada;
- TTL e cota validados;
- aprendizagem ainda desligada ou sob fluxo manual;
- dados simulados desabilitados;
- TLS, conta corporativa e controle de acesso implantados;
- responsável técnico e DPO aprovam o piloto.

Depois:

```text
RAG_AUDIT_VISION_MODE=entry
RAG_AUDIT_VISION_DRY_RUN=false
RAG_AUDIT_SEED_DEMO_DATA=false
```

Ativar primeiro para uma janela pequena, com operador acompanhando os eventos. Não ativar alertas disciplinares.

## 23. Casos de teste obrigatórios

| Caso | Resultado esperado |
|---|---|
| pessoa cadastrada entra | uma entrada com identidade correta |
| mesma pessoa permanece visível | nenhum novo evento |
| pessoa sai | não contar como entrada |
| pessoa se aproxima e recua | nenhuma entrada |
| cruzamento fora do segmento | nenhuma entrada |
| oscilação na linha | nenhuma entrada |
| desconhecido entra | `UNKNOWN`, sem nome forçado |
| dois candidatos próximos | `AMBIGUOUS` |
| rosto pequeno ou borrado | `LOW_QUALITY` |
| duas pessoas entram | duas trilhas, sem troca de identidade |
| track fragmenta | no máximo uma entrada por visita |
| identidade muda na trilha | consenso e trajetória reiniciados |
| API fora do ar | evento preservado na outbox |
| câmera fora e API volta | outbox continua enviando |
| evidência alterada | leitura bloqueada por integridade |
| evidência expira | arquivo e índice removidos |
| candidato aprendido | permanece `PENDING` |
| candidato rejeitado | nunca entra na busca |
| referência revogada | deixa de ser carregada |
| foto ou tela | não tratar como liveness |
| modo `door` sem override | somente dry-run |

## 24. Critérios de aceite

O piloto só avança quando:

- nenhuma saída controlada é descrita como entrada;
- nenhum nome é atribuído abaixo dos limites aprovados;
- FPIR, FNIR e rank-1 são medidos em validação separada;
- duplicidade e fragmentação estão dentro do limite;
- falhas de API não perdem evento persistido;
- falha de câmera não bloqueia a outbox;
- credenciais não aparecem em logs;
- imagens não aparecem no webhook, PDF ou alerta;
- evidências vencem e são expurgadas;
- candidatos aprendidos exigem ação humana;
- operadores conseguem revisar e contestar um evento;
- responsáveis aprovam a matriz de risco.

Não existe limiar universal. Se o conjunto de validação não tiver desconhecidos e pares difíceis, não há base para aprovar FPIR.

## 25. Observabilidade

Registrar métricas separadas:

- FPS recebido e processado;
- tempo de detecção;
- tempo de embedding e ranking;
- quantidade de faces por estado;
- reconexões RTSP;
- idade e tamanho da outbox;
- tentativas e falhas HTTP;
- uso de CPU, memória e disco;
- cota de evidências;
- atraso câmera → worker;
- atraso worker → API;
- latência ponta a ponta.

Logs devem usar IDs técnicos. Nome, imagem, embedding, senha e URL RTSP não pertencem ao log operacional.

## 26. Segurança e privacidade

Biometria é dado pessoal sensível. Antes de usar a base real:

- documentar finalidade, necessidade e hipótese legal;
- produzir ou atualizar o RIPD;
- informar as pessoas;
- aplicar menor privilégio;
- usar MFA na administração;
- criptografar disco, backup e transporte;
- definir retenção e descarte;
- separar galeria, evidências e banco da API;
- revisar acesso administrativo;
- manter resposta a incidente;
- oferecer correção e contestação;
- proibir decisão exclusivamente automatizada.

O bundle de modelos também possui uma condição de uso própria. A presença do script de preparação não concede licença sobre os pesos.

## 27. Rollback

Se o piloto falhar:

1. voltar `RAG_AUDIT_VISION_DRY_RUN=true`;
2. parar o worker;
3. desabilitar o destinatário de e-mail, se configurado;
4. manter a fechadura independente;
5. preservar somente logs técnicos necessários;
6. exportar métricas sem imagens;
7. revogar referências aprendidas do período, se houver dúvida;
8. expurgar evidências conforme a política;
9. restaurar configuração da câmera;
10. registrar causa, impacto e condição para novo teste.

## 28. Checklist final

### Ambiente

- [ ] modelo e firmware registrados;
- [ ] IP, VLAN, ACL e NTP validados;
- [ ] usuário RTSP exclusivo;
- [ ] servidor protegido e sem suspensão;
- [ ] disco criptografado;
- [ ] segredos fora do Git.

### Imagem

- [ ] stream principal estável;
- [ ] rosto iluminado na soleira;
- [ ] IED medido em alturas e posições diferentes;
- [ ] pose, nitidez e oclusão medidas;
- [ ] ROI definida;
- [ ] zonas e linha testadas em ambos os sentidos.

### Modelos e galeria

- [ ] licença conferida;
- [ ] bundle preparado;
- [ ] fingerprint fixado;
- [ ] hash verificado no startup;
- [ ] galeria pequena e autorizada;
- [ ] fotos e IDs conferidos;
- [ ] referências antigas incompatíveis bloqueadas.

### Reconhecimento

- [ ] consenso conservador;
- [ ] margem top-1/top-2 calibrada;
- [ ] FPIR medido;
- [ ] FNIR medido;
- [ ] rank-1 medido;
- [ ] desconhecidos e ambiguidades revisáveis;
- [ ] fragmentação e troca de identidade testadas.

### Operação

- [ ] modo `entry`;
- [ ] dry-run concluído;
- [ ] modo `door` restrito;
- [ ] outbox testada com API e câmera fora;
- [ ] evidência com TTL, cota e integridade;
- [ ] aprendizagem desativada ou manual;
- [ ] alertas sem decisão automática;
- [ ] rollback ensaiado.

### Governança

- [ ] finalidade e hipótese legal documentadas;
- [ ] RIPD avaliado;
- [ ] retenção aprovada;
- [ ] revisão humana definida;
- [ ] contestação definida;
- [ ] resposta a incidente definida;
- [ ] aceite técnico e de privacidade registrado.

## 29. Referências

- [Manual VIPC 1230 B/D G2](https://backend.intelbras.com/sites/default/files/2026-04/Manual_VIPC_1230_BD_G2_02-26_site%20v7.pdf)
- [Ficha técnica VIPC 1230 B/D G2](https://backend.intelbras.com/sites/default/files/2026-02/Datasheet%20UNIFICADO%20-%20VIPC%201230%20BD%20G2%20v7.pdf)
- [Manual InControl Web](https://manual-incontrol.intelbras.com.br/pt-BR/manual_pt-BR.html)
- [InsightFace](https://github.com/deepinsight/insightface)
- [Lei Geral de Proteção de Dados](https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709compilado.htm)
