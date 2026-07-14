# Plano de implantação no ambiente Intelbras

## 1. Objetivo

Conectar a câmera **Intelbras VIPC 1230 B/D G2**, a base autorizada de pessoas e o
RAG-Audit para produzir um registro auditável quando uma pessoa for observada entrando
na sala.

O sistema não controla a fechadura e a fechadura por senha não fornece log. Portanto,
o resultado correto é **entrada observada visualmente**, nunca “senha aceita” ou
“porta liberada”. Uma decisão disciplinar, trabalhista ou de bloqueio não pode ser
tomada automaticamente a partir do reconhecimento.

## 2. Decisão de arquitetura após a descoberta do e-mail

A câmera consegue enviar e-mail ao detectar movimento e anexar uma foto. O manual
oficial confirma servidor e porta SMTP configuráveis, modos Nenhum/SSL/TLS, até três
destinatários e a opção **Anexar foto**.

O e-mail será usado como **gatilho e evidência complementar**, não como prova isolada
de entrada. Uma foto única não informa com segurança se a pessoa entrou, saiu, ficou
parada na porta ou apenas apareceu na região de movimento.

O manual não documenta o formato MIME, nome/extensão/resolução do anexo, `Message-ID`,
retries, duplicidade nem cooldown dos e-mails de evento. Também não esclarece se “TLS”
é STARTTLS ou se “SSL” é TLS implícito. Esses detalhes serão medidos no firmware real,
sem presumir compatibilidade.

Arquitetura recomendada:

```text
                         +-----------------------------+
                         | Base Intelbras autorizada   |
                         | InControl / Defense /       |
                         | controlador facial          |
                         +--------------+--------------+
                                        |
                               IDs + fotos permitidas
                                        |
                                        v
+----------------+     SMTP      +------+------------------+
| VIPC 1230 G2   +-------------->| caixa IMAP dedicada ou |
| movimento/foto |               | valida MIME, origem,    |
+-------+--------+               | tamanho, hash e anexo   |
        |                        +------+------------------+
        | RTSP                          | gatilho
        |                               v
        |                 +-------------+------------------+
        +---------------->| worker local de visão          |
                          | YuNet + SFace + tracking        |
                          | porta -> linha IN -> interior   |
                          +-------------+------------------+
                                        |
                              evento normalizado, sem foto
                                        |
                                        v
                          +-------------+------------------+
                          | FastAPI RAG-Audit               |
                          | contexto, risco, log, alerta,   |
                          | dashboard, relatório e PDF      |
                          +---------------------------------+
```

### Modos possíveis

1. **Somente e-mail/foto:** identifica um rosto observado, mas o evento fica como
   `FACE_OBSERVED` ou `NO_DIRECTION_EVIDENCE`; não conta como entrada confirmada.
2. **E-mail + RTSP (recomendado):** o e-mail inicia/correlaciona a análise e o RTSP
   confirma a trajetória porta → interior. Somente então o evento entra nos indicadores.
3. **RTSP contínuo:** fallback quando o e-mail atrasar, perder anexo ou ficar indisponível.
4. **Sensor de porta futuro:** eleva a força da evidência sem mudar o reconhecimento.

Como o SMTP normalmente chega depois da captura, abrir o RTSP somente após receber o
e-mail pode perder o cruzamento. A evolução híbrida manterá o stream extra em baixa
carga e uma janela curta em memória; o e-mail ancora o evento e aciona análise de
maior qualidade no stream principal/anexo. O tamanho dessa janela será definido pela
latência medida no ambiente, não por suposição.

### 2.1 Opções para receber o alerta

**Primeiro piloto — caixa dedicada + IMAP/TLS:** como o envio de e-mail já funciona,
adicionamos um dos três destinatários permitidos pela câmera e o worker lê essa caixa
por IMAP TCP 993. Isso evita trocar o servidor SMTP atual ou abrir uma porta no host.
A caixa deve ser corporativa e exclusiva; não usar Gmail/conta pessoal. UID/UIDVALIDITY
serão usados para durabilidade, e o e-mail só será marcado como processado após o
gatilho ser persistido.

**Destino final — coletor/relay SMTP local:** mantém a foto dentro da rede. Requer
validar compatibilidade TLS/autenticação do firmware, certificado, firewall, fila e
monitoramento. Não pode funcionar como relay aberto. Se o SMTP atual também atende
outra finalidade, a migração deve preservar o destinatário existente ou encaminhar uma
cópia aprovada.

As duas opções produzirão o mesmo objeto interno `CameraTrigger`; trocar o transporte
não muda rastreamento, reconhecimento, regras ou relatórios.

## 3. O que já está implementado no repositório

- teste TCP e leitura segura de um frame RTSP, sem exibir a senha;
- importador defensivo do ZIP do InControl e de pastas de fotos;
- descarte de senhas, cartões e credenciais presentes no ZIP;
- pseudonimização de identificadores documentais;
- modelos oficiais OpenCV YuNet/SFace com SHA-256 fixado;
- comparação com limiar, margem entre primeiro/segundo candidato e rejeição de baixa qualidade;
- consenso da mesma identidade em vários frames;
- rastreamento de múltiplos rostos e confirmação porta → interior;
- estados `MATCHED`, `UNKNOWN`, `AMBIGUOUS` e `LOW_QUALITY`;
- fila SQLite persistente para reenviar eventos se a API estiver fora do ar;
- contrato que proíbe `door_result=GRANTED` quando a evidência é somente visual;
- registro da versão do modelo, qualidade, margem, `track_id` e confiança da trajetória;
- dashboard, regras, alertas e PDFs da fase anterior.

O receptor SMTP específico será finalizado depois de observar um arquivo `.eml`
original da câmera. Isso evita supor assunto, nomes de anexo, `Message-ID`, charset ou
estrutura MIME que podem variar com o firmware.

### 3.1 Lacunas assumidas antes da produção

- ainda não existe receptor SMTP/IMAP nem correlação e-mail ↔ RTSP;
- o worker atual processa um RTSP continuamente e não mantém buffer pré-evento;
- o tracker atual acompanha centróide de rosto, não o corpo inteiro; o piloto dirá se
  será necessário detector/tracker de pessoa e associação rosto ↔ track;
- armazenamento de evidência com TTL/expurgo ainda não foi implementado;
- o contrato ainda não possui `trigger_source`, `trigger_id`, atraso de correlação e
  referência opaca de evidência;
- no compose de laboratório, API e worker compartilham `./data` e o mesmo `.env`;
  produção deverá separar segredos e conceder a galeria somente ao worker;
- o dashboard usa HTTP Basic e porta 8000; produção exige proxy HTTPS, identidade
  corporativa, RBAC e MFA.

## 4. O que levantar antes da visita técnica

Não colocar senhas neste documento, em chamados ou em mensagens. As credenciais serão
digitadas somente no `.env` da máquina local durante a implantação.

### 4.1 Câmera

- foto da etiqueta, confirmando `VIPC 1230 B G2` ou `VIPC 1230 D G2`;
- IP atual, máscara, gateway, VLAN e MAC;
- versão de firmware e versão web, obtidas em **Informações → Versão**;
- portas HTTP, HTTPS e RTSP configuradas;
- resolução, codec e FPS dos streams principal e extra;
- estado do NTP, servidor NTP, fuso e horário exibido;
- captura de tela de **Serviços → SMTP (E-mail)**, ocultando usuário/senha;
- captura de tela de **Eventos → Detecção de movimento**, incluindo região,
  sensibilidade, limiar, período, estabilização, Enviar e-mail e Anexar foto;
- informar se a câmera está ligada diretamente a um NVR e o modelo do NVR;
- informar se existe contraluz, vidro, espelho ou outra porta no enquadramento.

### 4.2 Amostras do alerta da câmera

Salvar localmente, sem encaminhar por e-mail, os seguintes arquivos:

- um **e-mail de teste** exportado no formato original `.eml`;
- um alerta real de movimento com foto, usando somente um colaborador de teste
  informado e autorizado;
- dois alertas consecutivos do mesmo movimento, para testar deduplicação;
- a foto anexada exatamente como enviada pela câmera;
- horário mostrado pela câmera, horário do evento e horário de chegada do e-mail.

Encaminhar/“forward” altera cabeçalhos e MIME; por isso precisamos do `.eml` original.
Durante a sessão, ele pode ser colocado em:

```text
data/private/samples/camera-test.eml
data/private/samples/camera-motion.eml
```

Esses arquivos estão fora do Git e devem ser excluídos após a validação.

### 4.3 Base facial Intelbras

Identificar qual alternativa é usada:

- **InControl Web:** nome e versão do software e exportação ZIP de 3 a 5 usuários de
  teste. O ZIP contém CSV e fotos; não trazer o banco PostgreSQL completo.
- **Defense IA:** versão, tela de Swagger/API e dispositivo que executa a comparação.
- **Controlador facial independente:** modelo exato, firmware e software de cadastro.
- **Outro sistema:** nome, versão e forma oficial de exportar IDs e fotos.

Para o piloto, usar apenas pessoas informadas e autorizadas. Não copiar toda a base de
produção antes de validar finalidade, segurança e retenção.

### 4.4 Servidor que executará o projeto

- sistema operacional e versão;
- CPU, RAM, espaço livre e existência de GPU;
- IP/máscara/gateway/VLAN da máquina;
- BitLocker ou outra criptografia de disco ativa;
- política de backup e antivírus/EDR;
- conta de serviço que executará o worker;
- confirmação de que a máquina não entra em suspensão/hibernação;
- decisão entre execução Python local ou Docker.

Neste computador, o Docker Desktop não iniciou porque a virtualização AMD-V/SVM não
estava habilitada no firmware. O projeto funciona com Python local; para Docker será
necessário habilitar SVM/AMD-V e WSL2/Virtual Machine Platform, com reinicialização
planejada.

### 4.5 Aprovações e regras de negócio

- responsável técnico pelo ambiente e pela câmera;
- encarregado/DPO ou responsável de privacidade;
- finalidade documentada para uso da biometria na sala;
- lista inicial de pessoas que podem entrar na sala;
- horários/plantões e fonte dessas informações;
- quem revisará `UNKNOWN` e `AMBIGUOUS`;
- prazo de retenção de foto, evento e relatório;
- canal para contestação/correção de falso reconhecimento;
- canal real de alertas: Teams, Slack, Telegram, e-mail ou webhook interno.

## 5. Rede e portas

Fluxos mínimos sugeridos:

| Origem | Destino | Porta | Uso |
|---|---|---:|---|
| servidor RAG-Audit | câmera | TCP 554 | RTSP |
| câmera | receptor SMTP local/relay | TCP 587 ou porta aprovada | alerta com foto |
| worker de e-mail | caixa corporativa | TCP 993 | IMAP/TLS no primeiro piloto |
| câmera e servidor | NTP interno | UDP 123 | relógios sincronizados |
| estação administrativa | câmera | TCP 443 | configuração HTTPS |
| estação administrativa | RAG-Audit | TCP 8000 no piloto | dashboard/API |
| RAG-Audit | webhook corporativo | TCP 443 | alertas, se habilitados |
| dispositivos | DNS interno | UDP/TCP 53 | apenas se forem usados nomes |

Regras:

- não publicar RTSP, SMTP ou painel da câmera na internet;
- preferir VLAN de CFTV/IoT e ACL explícita entre câmera e servidor;
- criar usuário exclusivo de leitura para RTSP, sem reutilizar o administrador;
- usar TLS e autenticação no SMTP quando o firmware/relay permitirem;
- se o piloto usar SMTP anônimo, restringir por firewall ao IP/MAC/VLAN da câmera,
  usar destinatário exclusivo e migrar para relay autenticado antes de produção;
- o endereço `From` não autentica a câmera e pode ser falsificado;
- sincronizar câmera, servidor e sistema de controle de acesso no mesmo NTP.

O receptor também precisa distinguir movimento de outros e-mails possíveis da câmera,
como mascaramento, acesso ilegal, alerta de segurança e e-mail periódico de teste.

## 6. Configuração da câmera durante a sessão

Antes de alterar qualquer item, exportar configuração quando possível e registrar
capturas das telas atuais para rollback.

### 6.1 Vídeo

- stream principal: 1920×1080, H.264, qualidade adequada, usado para o rosto;
- stream extra: H.264 e resolução menor, opcional para tracking econômico;
- evitar H.265+ no primeiro teste para reduzir incompatibilidade/latência;
- ajustar exposição, DWDR/BLC e iluminação para evitar rosto escuro;
- posicionar a câmera de forma que o rosto atravesse uma região visível, não apenas
  cresça no centro do quadro;
- desabilitar áudio; este modelo não precisa enviar áudio para a aplicação.

### 6.2 Movimento e e-mail

- desenhar a região apenas na soleira/área imediatamente interna;
- começar com sensibilidade e limiar médios e medir falsos gatilhos;
- habilitar período integral apenas durante o teste controlado;
- habilitar **Enviar e-mail** e **Anexar foto**;
- usar assunto exclusivo, por exemplo `RAG-AUDIT CAM-TI-01 MOVIMENTO`;
- destinatário exclusivo por câmera;
- enviar e-mail de teste e depois um evento real;
- medir atraso e verificar se o anexo contém rosto suficiente.

O botão “E-mail de teste” valida SMTP, mas não valida movimento, foto, direção nem
reconhecimento. Todos esses itens precisam de testes separados.

Na tela de evento, a opção **Foto** é uma ação diferente, destinada ao FTP; ela não é
necessária para anexar a imagem ao e-mail. O campo **Atualizar período** pertence ao
e-mail periódico de teste e não deve ser confundido com intervalo/cooldown dos alertas
de movimento. O nome da região de movimento será configurado por câmera e usado como
mais um sinal de classificação, sem ser tratado como autenticação.

## 7. Procedimento técnico no servidor

Diretório atual:

```powershell
cd C:\caminho\para\rag-audit
```

### 7.1 Preencher segredos localmente

Editar `.env` e preencher somente na máquina:

```text
INTELBRAS_CAMERA_HOST=
INTELBRAS_CAMERA_USER=
INTELBRAS_CAMERA_PASSWORD=
```

Não compartilhar o conteúdo preenchido. O restante das chaves já possui valores de
laboratório documentados no `.env.example`.

### 7.2 Testar rede, credencial e RTSP

```powershell
.\.venv\Scripts\python.exe scripts\probe_intelbras_camera.py
```

Resultado esperado:

- porta 554 acessível;
- stream autenticado abre;
- um frame é lido na memória e descartado;
- nenhuma URL com usuário/senha aparece no terminal.

### 7.3 Importar somente a galeria de teste

Exemplo para InControl:

```powershell
.\.venv\Scripts\python.exe scripts\import_incontrol_gallery.py `
  C:\Temp\incontrol-teste.zip `
  data\private\gallery
```

O importador:

- bloqueia ZIP Slip, links, arquivos especiais e bombas de compressão;
- ignora senha, cartão e credenciais;
- detecta fotos mesmo sem extensão;
- grava somente manifesto mínimo e imagens necessárias;
- produz hash SHA-256 de cada foto.

Sincronizar IDs/nome no banco da auditoria:

```powershell
.\.venv\Scripts\python.exe scripts\sync_gallery_people.py
```

Essa etapa não concede automaticamente permissão para a sala nem cria horários.
Permissões precisam de uma decisão explícita do responsável.

### 7.4 Capturar quadro vazio para calibrar a porta

Com a sala e corredor vazios:

```powershell
.\.venv\Scripts\python.exe scripts\capture_calibration_frame.py --empty-room-confirmed
```

O quadro será salvo temporariamente em:

```text
data/private/camera-calibration-frame.jpg
```

Usaremos esse quadro para desenhar coordenadas normalizadas de:

- `door_zone`;
- `entry_line` e seu lado interno;
- `inside_zone`.

Depois será criado `data/private/camera-calibration.json` a partir de
`config/camera-calibration.example.json`, e o quadro temporário será excluído.

### 7.5 Validar toda a cadeia sem monitorar

```powershell
.\.venv\Scripts\python.exe scripts\run_vision_worker.py --check
```

Essa verificação cobre câmera, RTSP, modelos, galeria e geometria.

### 7.6 Dry-run

Manter:

```text
RAG_AUDIT_VISION_DRY_RUN=true
```

Executar:

```powershell
.\.venv\Scripts\python.exe scripts\run_vision_worker.py
```

No dry-run, eventos ficam na fila local e não entram no dashboard. Primeiro serão
ajustados direção, repetição, qualidade e limiares.

### 7.7 Ativar envio

Somente após aprovação dos testes:

```text
RAG_AUDIT_VISION_DRY_RUN=false
```

Reiniciar o worker. O dashboard permanece em:

```text
http://localhost:8000/dashboard
```

Antes do piloto real também será necessário desabilitar dados simulados:

```text
RAG_AUDIT_SEED_DEMO_DATA=false
```

Os eventos simulados existentes devem ser arquivados ou separados do banco real.

## 8. Contrato do evento final

Exemplo sem imagem/template:

```json
{
  "event_id": "entry:vis-...",
  "camera_id": "cam-ti-01",
  "user_id": "EMP001",
  "room_id": "sala_ti_01",
  "timestamp": "2026-07-14T18:00:00-03:00",
  "door_result": "NOT_REPORTED",
  "recognition_confidence": 0.84,
  "identity_status": "MATCHED",
  "entry_evidence": "VISION_LINE_CROSSING",
  "recognition_source": "LOCAL_SFACE",
  "track_id": "boot-id:face-7",
  "recognition_model": "opencv-yunet-2023mar+sface-2021dec",
  "recognition_margin": 0.18,
  "face_quality": 0.87,
  "entry_confidence": 0.93
}
```

Fotos, e-mails, templates, senhas e base64 não entram nesse webhook.

## 9. Tratamento do e-mail

O receptor deverá seguir estas regras:

- aceitar somente IP/VLAN da câmera ou relay autenticado;
- destinatário exclusivo e aleatório por câmera;
- tamanho máximo do e-mail e dos anexos;
- aceitar somente JPEG/PNG/WebP validado por assinatura binária, não por extensão;
- remover EXIF/GPS e limitar dimensões/pixels antes do reconhecimento;
- bloquear anexos compactados, executáveis, HTML ativo e múltiplas estruturas suspeitas;
- normalizar MIME/charset sem executar conteúdo;
- calcular hash do e-mail e anexo para idempotência;
- registrar horários de recebimento/processamento, câmera e hashes;
- não confiar apenas em `From`, `Subject` ou `Message-ID`;
- no IMAP, deduplicar por caixa + `UIDVALIDITY` + UID e só confirmar após persistir;
- manter a imagem apenas pelo período aprovado e em diretório privado/criptografado;
- excluir temporários inclusive em erro;
- usar o anexo como observação facial e o RTSP como confirmação direcional;
- sinalizar atraso, anexo ausente, baixa qualidade e duplicidade separadamente.

Um adaptador do firmware será escrito depois de analisar os `.eml` originais.

Estados planejados do gatilho:

```text
RECEIVED -> VALIDATED -> WAITING_RTSP -> CORRELATED -> EVENT_EMITTED
                 |            |              |
                 v            v              v
             REJECTED   RTSP_CONTEXT_MISSING NO_ENTRY_CONFIRMATION
```

Outros finais possíveis: `DUPLICATE`, `EXPIRED`, `LOW_QUALITY`, `UNKNOWN` e
`AMBIGUOUS`. E-mail de teste, saída e movimento sem cruzamento não entram nas métricas
nem no PDF de acessos.

Tabelas planejadas para essa correlação:

- `camera_triggers`: câmera, fonte, hashes de MIME/anexo, horários, região e estado;
- `entry_event_sources`: vínculo evento ↔ gatilho ↔ track e janela de correlação;
- `evidence_objects`: chave opaca, hash, tipo, tamanho e expiração, sem BLOB no SQLite.

## 10. Casos de teste obrigatórios

| Caso | Resultado esperado |
|---|---|
| pessoa autorizada entra | uma entrada, identidade correta |
| mesma pessoa permanece na sala | nenhuma nova entrada |
| pessoa sai | não contabilizar como entrada |
| pessoa aparece e recua | `NO_DIRECTION_EVIDENCE`, sem entrada |
| desconhecido entra | `UNKNOWN`, revisão humana |
| dois candidatos muito próximos | `AMBIGUOUS`, sem atribuir identidade |
| duas pessoas entram juntas | duas trilhas; sinalizar possível carona |
| e-mail duplicado | um único gatilho/evento |
| e-mail sem anexo | erro controlado, sem identidade |
| foto/rosto pequeno ou borrado | `LOW_QUALITY`, sem identidade |
| RTSP cai durante evento | reconexão; evento pendente não se perde |
| API reinicia | fila reenvia sem duplicar |
| relógio da câmera divergente | falha operacional visível |
| foto impressa/tela | não tratar como prova de vida |

## 11. Critérios de aceite do piloto

- nenhum evento de saída é descrito como entrada nos testes controlados;
- nenhum candidato abaixo do limiar/margem recebe um nome forçado;
- evento repetido não gera duplicidade;
- perda temporária da API não perde o evento já confirmado;
- credenciais não aparecem em logs, relatórios ou dashboard;
- imagens não aparecem no PDF por padrão;
- latência da câmera, SMTP, reconhecimento e API é medida separadamente;
- casos desconhecidos/ambíguos possuem revisão humana;
- retenção e descarte são demonstráveis;
- responsáveis aprovam matriz de falso positivo/falso negativo antes de produção.

Não existe limiar universal de SFace para este local. O valor inicial serve apenas para
laboratório e será calibrado com pessoas, iluminação e posição reais.

## 12. Segurança, privacidade e retenção

Biometria é dado pessoal sensível. Antes de usar a base de produção:

- documentar finalidade, necessidade e hipótese legal com encarregado/jurídico;
- elaborar/atualizar o RIPD;
- informar as pessoas afetadas;
- restringir acesso por função e usar MFA na administração final;
- criptografar disco, backup e transporte;
- definir retenção curta para fotos e maior apenas para metadados necessários;
- não usar reconhecimento como única base de punição ou acusação;
- oferecer revisão e correção;
- manter procedimento de incidente e descarte.

Valores de retenção não serão escolhidos pelo código: devem ser aprovados pelo
controlador/DPO e parametrizados depois.

## 13. Rollback

Se o piloto falhar:

1. parar o worker de visão e o receptor SMTP;
2. desabilitar Enviar e-mail/Anexar foto ou restaurar a configuração registrada;
3. manter a fechadura totalmente independente;
4. preservar somente logs técnicos necessários para diagnóstico;
5. excluir galeria, anexos e temporários conforme a regra aprovada;
6. documentar motivo, falso positivo/negativo e condição de repetição do teste.

## 14. O que não trazer ou fazer

- não enviar senha da câmera, SMTP ou InControl por mensagem;
- não anexar a base facial completa em chamado/chat;
- não encaminhar e-mail de alerta: exportar o `.eml` original localmente;
- não abrir RTSP/HTTP/SMTP da câmera na internet;
- não usar conta administrativa para o worker;
- não ativar alertas reais antes do dry-run;
- não chamar uma foto de movimento de “entrada confirmada”;
- não armazenar templates/fotos no SQLite do RAG-Audit.

## 15. Referências oficiais

- [Manual VIPC 1230 B/D G2](https://backend.intelbras.com/sites/default/files/2026-04/Manual_VIPC_1230_BD_G2_02-26_site%20v7.pdf)
- [Ficha técnica VIPC 1230 B/D G2](https://backend.intelbras.com/sites/default/files/2026-02/Datasheet%20UNIFICADO%20-%20VIPC%201230%20BD%20G2%20v7.pdf)
- [Manual InControl Web](https://manual-incontrol.intelbras.com.br/pt-BR/manual_pt-BR.html)
- [OpenCV YuNet/SFace](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html)
- [Lei Geral de Proteção de Dados](https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709compilado.htm)
