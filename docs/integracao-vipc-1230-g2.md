# Integração da Intelbras VIPC 1230 G2

Este documento define como a câmera fixa Intelbras VIPC 1230 B/D G2 será usada para observar a entrada da sala. Ele descreve o sinal que a câmera realmente fornece, o reconhecimento no worker local e os limites que precisam permanecer visíveis nos registros.

## Papel da câmera

A VIPC 1230 G2 fornece vídeo 2 MP por RTSP, dois streams, ONVIF e eventos básicos de movimento. Ela não executa o reconhecimento usado pelo RAG-Audit e não informa qual senha foi digitada na fechadura.

Streams usuais:

```text
rtsp://USUARIO:SENHA@IP:554/cam/realmonitor?channel=1&subtype=0
rtsp://USUARIO:SENHA@IP:554/cam/realmonitor?channel=1&subtype=1
```

- `subtype=0`: stream principal, preferido para detectar e reconhecer o rosto;
- `subtype=1`: stream de menor resolução, útil somente se os testes mostrarem IED suficiente;
- usuário e senha ficam em segredo local ou cofre;
- a URL completa nunca deve aparecer em logs;
- câmera e servidor precisam usar NTP e o mesmo fuso.

Como a câmera não será substituída, o trabalho de campo deve aproveitar o máximo da imagem disponível. A prioridade é fazer o rosto ocupar mais pixels na passagem pela porta, evitar contraluz e limitar a análise à região relevante.

Referências do fabricante:

- [Manual da VIPC 1230 B/D G2](https://backend.intelbras.com/sites/default/files/2026-04/Manual_VIPC_1230_BD_G2_02-26_site%20v7.pdf)
- [Ficha técnica da VIPC 1230 B/D G2](https://backend.intelbras.com/sites/default/files/2026-02/Datasheet%20UNIFICADO%20-%20VIPC%201230%20BD%20G2%20v7.pdf)
- [VIPC 1230 B G2](https://www.intelbras.com/pt-br/camera-ip-custo-beneficio-com-poe-vipc-1230-b-g2)
- [VIPC 1230 D G2](https://www.intelbras.com/pt-br/camera-ip-dome-custo-beneficio-com-poe-vipc-1230-d-g2)

## O que conta como entrada

O modo padrão do worker é `entry`. A mesma trilha facial precisa cumprir a sequência:

```text
zona da porta -> cruzamento da linha no sentido IN -> zona interna
```

O rastreador direcional também verifica:

- cruzamento dentro do segmento útil da linha;
- distância mínima percorrida;
- zona morta ao redor da linha;
- tempo máximo entre porta, cruzamento e interior;
- permanência mínima nas zonas;
- cooldown e deduplicação por visita.

Isso reduz eventos causados por rosto parado, jitter do tracker, pessoa que recua, passagem fora da porta ou movimento no sentido de saída.

O evento confirmado usa:

```text
entry_evidence=VISION_LINE_CROSSING
door_result=NOT_REPORTED
```

O relatório pode dizer “entrada observada pela câmera”. Não pode dizer “senha aceita”, “porta liberada” ou “acesso concedido pela fechadura”.

### Modo `door`

O modo `door` observa um rosto próximo da porta sem provar direção. Ele existe para diagnóstico, ambientes onde a geometria ainda não foi calibrada e análise do gatilho por foto.

Seu resultado é:

```text
entry_evidence=VISION_FACE_AT_DOOR
door_result=NOT_REPORTED
```

Por padrão, esse modo deve permanecer em dry-run. A emissão para a API exige um override explícito, registrado e aprovado. Mesmo com o override, o evento continua sendo presença próxima à porta, não entrada confirmada.

## Fluxo do worker

```text
RTSP
  -> recorte da ROI
  -> SCRFD-10GF detecta e alinha faces
  -> filtros de qualidade
  -> ArcFace gera embeddings
  -> ranking na galeria
  -> tracking com posição e aparência
  -> consenso temporal
  -> regra direcional de entrada
  -> evidência privada
  -> outbox persistente
  -> API de auditoria
```

O worker é um processo separado da API. A API recebe o resultado final e continua responsável pela validação do contrato, contexto, risco, log, alerta, dashboard e PDF.

## ROI e geometria

A região de interesse deve abranger:

- a aproximação imediata da porta;
- a linha de cruzamento;
- a primeira área interna onde a pessoa ainda está visível;
- uma margem pequena para não cortar o rosto durante o movimento.

Áreas profundas da sala, mesas, monitores, janelas e corredores sem relação com a passagem devem ficar fora da ROI sempre que possível. Isso diminui detecções concorrentes e libera processamento para usar o stream principal.

A calibração fica em arquivo privado, criado a partir de:

```text
config/camera-calibration.example.json
```

Coordenadas são normalizadas entre 0 e 1. O arquivo define `door_zone`, `inside_zone`, `entry_line`, lado interno, quantidade mínima de observações, timeout, cooldown, deadband, margem do segmento, deslocamento mínimo e tempo máximo de transição.

O loader rejeita polígono auto-intersectante, centro da zona da porta no lado interno, centro da zona interna no lado externo e linha cujo segmento não cubra o caminho entre as duas zonas.

Uma linha desenhada em um screenshot não basta. A direção deve ser testada com entradas e saídas reais, e os pontos precisam continuar válidos depois de qualquer ajuste físico da câmera.

## Motor facial

O projeto usa InsightFace `buffalo_l` em ONNX Runtime:

| Arquivo | Função |
|---|---|
| `det_10g.onnx` | SCRFD-10GF para detecção e landmarks |
| `w600k_r50.onnx` | ArcFace W600K R50 para embedding de 512 dimensões |

A decisão 1:N combina:

- similaridade do melhor candidato;
- margem entre primeiro e segundo;
- qualidade da face;
- repetição temporal;
- consistência consecutiva;
- identidade e aparência da trilha.

Estados possíveis:

- `MATCHED`: identidade aceita pelo conjunto de regras;
- `UNKNOWN`: nenhum candidato confiável;
- `AMBIGUOUS`: os melhores candidatos ficaram próximos demais;
- `LOW_QUALITY`: a captura não permite comparação segura.

O sistema deve preferir `UNKNOWN` a atribuir o nome errado.

## Bundle externo de modelos

Os pesos do `buffalo_l` não são versionados no Git nem baixados em runtime. O operador é responsável por:

1. obter os arquivos de uma fonte permitida;
2. confirmar que a licença é compatível com o uso;
3. registrar uma referência da autorização;
4. preparar o manifesto;
5. fixar o fingerprint aprovado no ambiente.

```powershell
python scripts\prepare_vision_models.py `
  --root data\vision-models `
  --license-reference "registro da licença ou aprovação" `
  --accept-model-license

python scripts\verify_vision_models.py `
  --root data\vision-models `
  --expected-fingerprint <fingerprint>
```

A verificação confere esquema do manifesto, arquivos permitidos, tamanho, SHA-256, fingerprint e presença dos dois modelos obrigatórios. Links simbólicos, arquivos extras ou conteúdo alterado são rejeitados.

Uma troca de modelo não é uma atualização transparente. Ela exige nova avaliação e invalida o uso automático de referências aprendidas ligadas ao fingerprint anterior.

## Qualidade da face

A câmera tem resolução e ângulo fixos. Antes de ajustar similaridade, é necessário medir a qualidade física da captura.

### IED

IED é a distância entre os centros dos olhos na imagem. Ela indica quantos pixels úteis existem para representar o rosto. Um rosto detectado pode ser grande o suficiente para o detector e ainda pequeno demais para identificação confiável.

O piloto deve registrar a distribuição de IED na soleira, separando:

- pessoas mais baixas e mais altas;
- centro e bordas da passagem;
- entrada lenta e rápida;
- stream principal e extra;
- dia, noite e contraluz.

O limite final deve vir dessas amostras, não de um número copiado de outro projeto.

### Pose, nitidez e iluminação

Também são avaliados:

- yaw: rosto virado para o lado;
- pitch: câmera alta demais ou pessoa olhando para baixo;
- roll: inclinação;
- nitidez: movimento, compressão e foco;
- iluminação: rosto escuro, estourado ou sem contraste;
- corte e oclusão: boné, cabelo, máscara, batente ou borda da ROI.

Se a maior parte dos frames for rejeitada, reduzir o limiar de identidade não resolve a causa. Primeiro devem ser corrigidos enquadramento, exposição, luz e ROI.

## Tracking e consenso

O rastreador associa faces usando posição, tamanho, sobreposição, velocidade e aparência. Isso é importante quando duas pessoas passam juntas ou quando uma detecção desaparece por poucos frames.

Cada trilha possui seu próprio histórico de identidade. O consenso considera matches, desconhecidos, ambiguidades e baixa qualidade. Para confirmar um nome, exige maioria suficiente e observações consecutivas.

Um frame `LOW_QUALITY` não atualiza a aparência da trilha e também quebra a sequência de identidade recente. Pessoas desconhecidas de tracks diferentes não são fundidas por semelhança facial; em auditoria, um evento duplicado é preferível a ocultar a entrada de outra pessoa.

Se a aparência indicar que a trilha mudou de pessoa:

- o consenso anterior é descartado;
- os melhores frames são separados por identidade;
- a geometria de entrada daquele track é reiniciada;
- uma amostra antiga não pode ser aprendida em nome da pessoa nova.

Uma identidade já registrada como presente também precisa sobreviver à fragmentação de track pelo período de deduplicação. Um novo número de track não significa uma nova entrada.

## Base facial Intelbras

O método depende do sistema onde as pessoas foram cadastradas:

1. **InControl Web:** importar uma exportação oficial contendo IDs e fotos.
2. **Defense IA:** preferir API/Swagger suportado, preservando o ID externo.
3. **Controlador facial:** usar a API, SDK ou exportação do modelo exato.
4. **Outro sistema:** documentar fabricante, versão, formato e autorização.

O importador deve receber fotos, não templates biométricos proprietários. Um template Intelbras não pode ser tratado como JPEG ou embedding ArcFace.

Para InControl:

```powershell
python scripts\import_incontrol_gallery.py <origem> data\private\gallery
python scripts\sync_gallery_people.py `
  --database data\api\rag_audit.db `
  data\private\gallery\manifest.json
```

O ZIP deve ser limitado ao grupo autorizado para o piloto. O importador descarta senhas, cartões e outros campos que não pertencem à finalidade.

Importar pessoas não cria autorização. O contexto operacional deve ser aplicado por `scripts/provision_access_context.py`, usando uma cópia privada de `config/access-context.example.json`. O manifesto reúne salas, câmeras, pessoas, permissões, escalas e o conjunto completo de políticas, sem qualquer credencial.

```powershell
python scripts\provision_access_context.py `
  data\private\config\access-context.json `
  --database data\api\rag_audit.db `
  --dry-run `
  --replace-assignments
```

Depois de revisar a simulação e fazer backup do banco, repita sem `--dry-run`. O caminho acima é o caminho no host; `/app/data/api/rag_audit.db` existe somente dentro do container. `--replace-assignments` é necessário quando a retirada de uma permissão ou escala deve ser refletida no banco.

## Aprendizagem sob revisão

Aprendizagem automática fica desligada inicialmente. Depois de uma entrada bem reconhecida, uma referência adicional só pode ser proposta quando passar por regras mais fortes de qualidade, similaridade, margem, consenso e proveniência.

Mesmo assim, a referência nasce como `PENDING` e não participa do reconhecimento. As ações administrativas são:

- aprovar;
- rejeitar;
- revogar uma aprovação;
- listar a origem, qualidade, modelo, fingerprint e evidência associada.

Somente referências `APPROVED`, de pessoas ainda ativas e compatíveis com o modelo carregado, entram na galeria de execução. Aprovações e revogações são recarregadas no intervalo configurado, sem reiniciar o worker. Se o banco aprendido estiver indisponível ou adulterado, o motor continua somente com a galeria oficial.

## Evidência e privacidade

A evidência privada é opcional. Quando gravada, contém uma cena e, se disponível, uma miniatura. O nome do arquivo é uma referência aleatória e não revela pessoa ou data.

O `EvidenceStore` mantém índice próprio com:

- hash SHA-256;
- tamanho;
- tipo de mídia;
- criação e expiração;
- TTL;
- cota total;
- limite individual;
- verificação de integridade;
- expurgo.

O webhook leva apenas `evidence_ref`. Fotos e embeddings não entram no banco principal, no PDF ou no alerta externo. O endpoint administrativo deve ler a evidência por essa referência e negar arquivo expirado, ausente ou adulterado.

API e worker compartilham o mesmo índice e precisam usar exatamente a mesma política de TTL, cota, limite individual e descarte. O índice grava esses valores e recusa inicialização divergente. No dry-run, a gravação de foto fica desativada por padrão.

## Outbox e falhas

Um evento confirmado é persistido na outbox antes do envio HTTP. A API pode estar offline sem perder o evento. O reenvio usa o mesmo `event_id`, aproveitando a idempotência do webhook.

Cada câmera possui dois bancos distintos:

```text
data/private/outbox/cam-ti-01.db
data/private/outbox/cam-ti-01.dry-run.db
```

O primeiro é a fila de produção e também guarda o lease que impede dois workers para a mesma câmera. O segundo recebe apenas eventos simulados. Nunca copie itens do dry-run para produção. Itens transitórios usam `RETRYING`; respostas 4xx permanentes usam `DEAD`. A API limita a idade de eventos autenticados vindos da fila por `RAG_AUDIT_QUEUED_EVENT_MAX_AGE_SECONDS`.

A fila de dry-run possui retenção e quantidade máximas próprias. Ela nunca é drenada para a API. A fila de produção não descarta eventos pendentes automaticamente; tamanho, idade e espaço em disco precisam ser monitorados.

O loop de envio deve continuar ativo quando:

- o RTSP estiver desconectado;
- a câmera estiver reiniciando;
- não houver faces no frame;
- o worker estiver tentando reconectar.

Ao perder o stream, o worker fecha estados incompletos com segurança, preserva itens já confirmados e continua drenando a fila. Não deve criar entrada apenas porque a conexão caiu.

Consulta operacional sem expor o payload:

```powershell
python scripts\manage_vision_outbox.py `
  --database data\private\outbox\cam-ti-01.db `
  summary
python scripts\manage_vision_outbox.py `
  --database data\private\outbox\cam-ti-01.db `
  list --status RETRYING --status DEAD
```

Reenfileirar um `DEAD` e excluir um `SENT` ou `DEAD` exigem operador, motivo e confirmação exata do `event_id`. A ação administrativa fica registrada no banco da outbox.

Um banco legado `vision-outbox.db` não deve ser renomeado às cegas. Pare o worker, revise e resolva o backlog, confirme que os eventos pertencem à câmera correta e só então migre o arquivo para `outbox/cam-ti-01.db`. Um backlog de dry-run deve ser descartado ou arquivado conforme a política, nunca promovido.

## E-mail com foto

O alerta SMTP da câmera pode servir como gatilho complementar, mas ainda depende da análise de um `.eml` original do firmware usado no local.

Uma foto isolada pode mostrar:

- uma pessoa entrando;
- uma pessoa saindo;
- alguém parado;
- movimento sem passagem;
- um rosto fora do momento de cruzamento.

Portanto:

- e-mail sem RTSP gera no máximo observação de porta;
- RTSP direcional continua sendo a fonte da entrada;
- MIME, anexos, limites, hashes, duplicidade e atraso precisam ser validados;
- `From`, assunto e `Message-ID` não autenticam a câmera sozinhos;
- o anexo deve ser validado por conteúdo, limitado e descartado pelo TTL aprovado.

O desenho detalhado está no [plano de implantação](plano-implantacao-ambiente-intelbras.md).

## Docker

Os serviços usam arquivos de ambiente separados:

```text
.env.api       credenciais administrativas, chaves por câmera e alertas
.env.vision    RTSP, chave da câmera correspondente e parâmetros de visão
```

Crie-os a partir de `.env.api.example` e `.env.vision.example`. O Compose não injeta um arquivo no outro. A opção `--env-file .env.vision` também é usada nos comandos para resolver o caminho host do bundle de modelos.

Os containers da API e da visão executam como `10001:10001`, com raiz somente leitura, capabilities removidas e `no-new-privileges`. O worker também possui:

- modelo montado como somente leitura;
- `tmpfs` limitado para temporários.

O Compose entrega à API apenas seu banco e as evidências compartilhadas. Galeria oficial e calibração são mounts separados e somente leitura. Cache derivado, referências aprendidas, outbox e evidências possuem mounts graváveis próprios. `RAG_AUDIT_GALLERY_CACHE_PATH` inclui `{camera_id}` e evita recalcular toda a galeria após cada reinício sem liberar escrita sobre as fotos oficiais. O cache pode ser descartado e reconstruído. Os diretórios graváveis precisam pertencer ao UID/GID esperado no host Linux e ter backup e criptografia compatíveis com a política aprovada.

O campo `recognition_threshold` cadastrado para a câmera continua existindo para fontes externas. Eventos `LOCAL_ARCFACE` já passaram pelos limiares calibrados e pelo consenso no worker; a API não aplica novamente esse limiar legado e exige o nome do motor e o fingerprint SHA-256 do bundle verificado.

## Calibração e métricas

A avaliação deve usar amostras do mesmo ponto de instalação e separar ajuste de validação. As métricas mínimas são:

- FPIR: identificação incorreta em busca 1:N;
- FNIR: pessoa cadastrada não identificada;
- rank-1: identidade correta em primeiro lugar;
- taxa de desconhecidos, ambiguidades e baixa qualidade;
- entradas duplicadas;
- entradas perdidas;
- erro de direção;
- latência da captura até a persistência.

O conjunto precisa conter pessoas cadastradas e não cadastradas, inclusive pares visualmente semelhantes quando autorizado. Também deve variar altura, pose, luz, velocidade, acessórios e quantidade de pessoas.

O avaliador frame-level registra FPIR operacional e FPIR condicionado às imagens que chegaram à comparação, além da configuração do detector, providers, política de qualidade e ROI opcional. Múltiplas faces continuam sendo falha de aquisição nesse relatório e precisam de teste temporal separado.

O limiar de produção só pode ser aprovado depois de observar FPIR. Avaliar apenas fotos da própria galeria mede memorização, não o desempenho da porta.

`scripts/evaluate_vision_dataset.py` é uma avaliação frame a frame, com exatamente uma face por probe e ROI normalizada opcional. Ela ajuda a comparar limiares, mas não exercita tracking, consenso, deduplicação ou direção. O aceite precisa combinar esse resultado com testes temporais filmados na porta.

## Limitações explícitas

- não há prova da senha digitada;
- não há leitura do estado da fechadura;
- não há contato magnético da porta integrado;
- a câmera RGB não fornece liveness validado;
- uma foto, tela ou máscara pode enganar o reconhecimento;
- uma classificação do app não é prova disciplinar;
- mudanças de luz, posição ou firmware podem exigir nova calibração.

O uso deve permanecer voltado a auditoria, investigação e revisão humana. Qualquer automação de bloqueio, punição ou concessão física de acesso está fora do escopo.

## Checklist técnico resumido

- [ ] confirmar modelo B/D G2 e firmware;
- [ ] criar usuário RTSP exclusivo de leitura;
- [ ] fixar IP, NTP, fuso, codec, resolução e FPS;
- [ ] ajustar enquadramento, luz, foco, exposição e ROI;
- [ ] obter bundle licenciado e fixar seu fingerprint;
- [ ] importar somente a galeria autorizada;
- [ ] calibrar zonas, linha e sentido;
- [ ] medir IED, pose, nitidez e iluminação;
- [ ] executar dry-run em modo `entry`;
- [ ] medir FPIR, FNIR, rank-1, duplicidade, direção e latência;
- [ ] validar outbox com API e câmera indisponíveis;
- [ ] validar TTL, cota, integridade e exclusão das evidências;
- [ ] revisar candidatos `PENDING` sem ativação automática;
- [ ] aprovar retenção, revisão e resposta a incidentes;
- [ ] só então considerar o envio de eventos reais.
