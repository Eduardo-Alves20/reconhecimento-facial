# Integração da Intelbras VIPC 1230 G2

Este documento descreve a próxima fase do RAG-Audit para uma câmera **VIPC 1230 B G2** ou
**VIPC 1230 D G2** instalada dentro da sala e apontada para a porta.

## O que o equipamento fornece

A VIPC 1230 G2 é a fonte de imagem. A ficha técnica documenta vídeo 2 MP, dois streams RTSP,
ONVIF e detecção simples de movimento. Ela não oferece detecção/classificação de pessoas,
reconhecimento facial nem cruzamento direcional de linha. Portanto, um worker local precisa
detectar, acompanhar e comparar os rostos. A API principal recebe apenas o resultado normalizado;
ela não deve processar todos os frames.

O envio SMTP com uma foto de movimento será usado como gatilho complementar. Ele não substitui
a confirmação direcional do RTSP, porque uma imagem isolada não distingue entrada, saída ou
permanência. O roteiro de campo e a arquitetura híbrida estão no
[plano de implantação](plano-implantacao-ambiente-intelbras.md).

Streams documentados pelo fabricante:

```text
rtsp://USUARIO:SENHA@IP:554/cam/realmonitor?channel=1&subtype=0
rtsp://USUARIO:SENHA@IP:554/cam/realmonitor?channel=1&subtype=1
```

- `subtype=0`: stream principal, usado para selecionar o melhor quadro do rosto.
- `subtype=1`: stream extra, que pode ser usado para detecção/tracking com menor custo.
- As credenciais ficam em segredo local ou cofre; nunca entram no Git, no dashboard ou em logs.
- A câmera e o servidor devem usar NTP e o mesmo fuso.

Referências oficiais:

- [Manual da VIPC 1230 B/D G2](https://backend.intelbras.com/sites/default/files/2026-04/Manual_VIPC_1230_BD_G2_02-26_site%20v7.pdf)
- [Ficha técnica da VIPC 1230 B/D G2](https://backend.intelbras.com/sites/default/files/2026-02/Datasheet%20UNIFICADO%20-%20VIPC%201230%20BD%20G2%20v7.pdf)
- [Página da VIPC 1230 B G2](https://www.intelbras.com/pt-br/camera-ip-custo-beneficio-com-poe-vipc-1230-b-g2)
- [Página da VIPC 1230 D G2](https://www.intelbras.com/pt-br/camera-ip-dome-custo-beneficio-com-poe-vipc-1230-d-g2)

## Quando o sistema considera que houve entrada

A fechadura por senha não fornece log. Por isso, a câmera não pode declarar que a porta liberou
o acesso. Ela produz uma **entrada observada visualmente** somente quando o mesmo track percorre:

```text
zona da porta -> cruza a linha no sentido IN -> zona interna -> permanência mínima
```

Regras importantes:

- rosto visto sem cruzamento não é entrada;
- trajetória da zona interna para a porta é saída;
- pessoa parada, oscilação na linha e reflexo não geram evento final;
- cada track confirmado gera no máximo uma entrada durante o período de deduplicação;
- duas pessoas cruzando após uma única abertura são duas entradas e podem sinalizar possível carona;
- o evento final usa `door_result=NOT_REPORTED`, nunca `GRANTED`;
- o texto do relatório deve dizer “entrada observada pela câmera”, não “entrada liberada”.

Sem contato magnético, sensor ou log da fechadura, essa evidência é probabilística. Um sensor de
porta barato pode ser acrescentado depois para elevar a confirmação sem substituir a câmera.

## Fluxo técnico

```text
VIPC 1230 G2 (RTSP)
  -> detector de pessoa/rosto
  -> tracker e linha direcional
  -> seleção do melhor quadro
  -> comparação com galeria autorizada
  -> correlação da identidade com a entrada
  -> evento normalizado
  -> regras, log, alerta, dashboard e PDF já existentes
```

O componente de visão deve ser um processo/container separado (`intelbras-worker`). O FastAPI
continua sendo a única camada que grava os eventos finais no SQLite. Isso impede que frames ou
observações intermediárias poluam métricas, PDFs e alertas.

## Motor facial do piloto

Para um primeiro piloto local, a opção recomendada é OpenCV com **YuNet** para localizar/alinhar
o rosto e **SFace** para gerar e comparar embeddings. Essa combinação funciona em CPU, Windows e
container sem TensorFlow. Os modelos precisam ser fixados por versão e SHA-256; produção não deve
baixar pesos automaticamente.

Referências oficiais:

- [Tutorial FaceDetectorYN e FaceRecognizerSF](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html)
- [YuNet no OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- [SFace no OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)

O limiar publicado em benchmark não deve ser copiado como regra de produção. Este caso é busca
1:N em uma porta real; exige calibração local, qualidade mínima, margem entre o primeiro e o segundo
candidato e confirmação da mesma identidade em vários frames. YuNet/SFace também não fornecem
prova de vida: uma foto ou tela pode enganar uma câmera RGB, então o resultado serve para auditoria
e revisão humana, não para punição ou bloqueio automático.

## Base facial Intelbras

O caminho de sincronização depende de onde as pessoas estão cadastradas:

1. **InControl Web:** usar a exportação ZIP de usuários. Segundo o manual, ela contém o CSV, as
   credenciais e as fotos de perfil. PDF e CSV isolados não levam as imagens. O RAG-Audit deve
   importar apenas identificador, nome e fotos necessárias, sem ler diretamente o PostgreSQL do
   InControl.
2. **Defense IA:** preferir a API oficial/Swagger e manter o identificador externo do usuário.
3. **Controlador facial independente:** usar o SDK/API do modelo exato ou uma exportação oficial.

Referência oficial do primeiro caminho:

- [Manual do InControl Web — importar e exportar usuários](https://manual-incontrol.intelbras.com.br/pt-BR/manual_pt-BR.html#exportando-dados)

Nunca tentar reutilizar um template biométrico proprietário como se fosse uma foto. Se a solução
Intelbras não fornecer imagens ou uma API de comparação suportada, a integração precisa ser feita
com o fabricante/integrador ou com um novo cadastro autorizado.

## Estados do reconhecimento

- `MATCHED`: candidato acima do limiar calibrado e com separação suficiente do segundo colocado.
- `UNKNOWN`: nenhum candidato confiável; gera evento com ID desconhecido exclusivo da sessão.
- `AMBIGUOUS`: candidatos próximos; encaminha para revisão humana e não atribui identidade.
- `LOW_QUALITY`: rosto pequeno, borrado, de perfil ou obstruído; não atribui identidade.

Uma porcentagem genérica não deve ser tratada como verdade. O limiar, a qualidade mínima e a
margem entre os dois melhores candidatos precisam ser calibrados com imagens reais da porta.

## Dados e evidências

O webhook atual armazena o payload integral. Portanto, imagens, templates e base64 nunca devem ser
enviados por ele. Evidências opcionais ficam em armazenamento privado com chave opaca, hash,
controle de acesso e expiração. O banco guarda somente as referências necessárias à auditoria.

Tabelas previstas para esta fase:

- `external_identities`: vínculo entre ID Intelbras e pessoa do RAG-Audit;
- `face_observations`: melhor observação do track, estado, qualidade, modelo e candidato;
- `entry_sessions`: direção, cruzamento, persistência e evidência visual;
- `evidence_objects`: chave, hash, tipo e data de descarte, sem BLOB no SQLite.

## Piloto recomendado

1. Fixar IP da câmera, criar usuário exclusivo de leitura, habilitar H.264 e sincronizar NTP.
2. Posicionar a câmera para que o rosto fique frontal e iluminado ao cruzar a soleira. Evitar
   contraluz da porta; ajustar DWDR/BLC se necessário.
3. Identificar o software ou controlador que contém a base facial.
4. Importar somente um grupo pequeno e autorizado de teste.
5. Marcar a zona da porta, linha IN e zona interna em um frame real.
6. Testar entrada, saída, duas pessoas juntas, boné/óculos, baixa luz e desconhecido.
7. Medir falso positivo, falso negativo, duplicidade e latência antes de habilitar alertas reais.

Biometria é dado pessoal sensível. O piloto exige finalidade documentada, acesso restrito, retenção
curta, aviso às pessoas, revisão humana e validação do encarregado/jurídico. A classificação do app
não deve aplicar punição ou negar direitos automaticamente.
