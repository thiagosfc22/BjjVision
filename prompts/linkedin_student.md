# Tarefa: imagem de post no LinkedIn — modelo aluno do BjjVision

Você está numa sessão nova, sem contexto anterior. Tudo o que precisa está aqui.
O repositório é `~/Documents/projetos/BjjVision` (branch `dalpra-dorsey`).
Rode tudo com `.venv/bin/python` (tem torch 2.13 com MPS, cv2, PIL, numpy).

## O que gerar

**Uma imagem PNG de 1200 × 1500** (proporção 4:5, o formato de maior alcance no
feed do LinkedIn), salva em `data/out/linkedin_student.png`, gerada por um script
novo em `scripts/make_linkedin_student.py`.

**Não invente nenhum número.** Todos os valores abaixo foram medidos. O script
deve ler os que puder em tempo de execução (contagem de parâmetros via forward
pass, tamanhos via `os.path.getsize`) em vez de digitar — assim o post continua
verdadeiro depois da próxima mudança no modelo.

## O que a imagem precisa comunicar em 2 segundos

Duas pessoas vão bater o olho e as duas precisam entender:

- **um recrutador**, que não sabe o que é IoU: "isto é um projeto sério de visão
  computacional, aplicado a jiu-jitsu brasileiro, e a pessoa treinou o modelo".
- **um técnico**, que sabe: "U-Net de 3,35M parâmetros destilando SAM2, com
  número de validação honesto e split por shot".

Por isso a imagem **precisa mostrar uma cena real de jiu-jitsu** — quimono,
tatame, faixa, placar da IBJJF. Nada de diagrama abstrato sozinho. O par
entrada → saída é o coração do post.

## Hierarquia visual, em ordem de peso

1. **O par entrada → saída** (maior elemento). Lado a lado: à esquerda o frame
   cru da luta, à direita o mesmo frame com as duas máscaras coloridas.
   Uma seta entre os dois. Rótulos: `entrada · 1 frame` e `saída · 2 máscaras`.
2. **O número de parâmetros**, grande e sem abreviar: `3.350.339`. Este é o
   destaque pedido — deve competir em peso com o hero, não ser uma nota de
   rodapé. Legenda: `parâmetros treinados do zero` (é literal: sem backbone
   pré-treinado, inicialização aleatória).
3. **Uma tira de thumbnails** com 4 frames adicionais já mascarados, provando
   que funciona em posições diferentes.
4. **Uma linha de métricas** compacta: IoU, fps, tamanho, comparação com SAM2.
5. **Assinatura** no rodapé.

Se não couber tudo com folga, corte na ordem inversa. Espaço em branco vale mais
que um item a mais — as peças anteriores desse projeto são deliberadamente arejadas.

## Assets e como renderizar

| coisa | caminho |
|---|---|
| vídeo fonte (1280×720, 30 fps, 23.306 frames) | `data/interim/galvao-xande_norm.mp4` |
| dataset do aluno (memmap 320×180 + manifest) | `data/out/student_gx_320/` |
| checkpoint treinado | `data/out/student_ckpt_v1/student.pt` |
| modelo | `src/bjjvision/student.py` → `UNetStudent`, `normalise` |
| leitor do dataset | `src/bjjvision/studentdata.py` → `StudentSet` |

Para prever a máscara de um frame:

```python
import sys; sys.path.insert(0, "src")
import cv2, numpy as np, torch
from bjjvision.student import UNetStudent, normalise
from bjjvision.studentdata import StudentSet

dev = torch.device("mps")
ck = torch.load("data/out/student_ckpt_v1/student.pt", map_location="cpu", weights_only=False)
model = UNetStudent(ck["width"]).to(dev); model.load_state_dict(ck["model"]); model.eval()

ds = StudentSet("data/out/student_gx_320")
pos = {int(f): i for i, f in enumerate(ds.frames)}     # nº do frame no vídeo -> índice

i = pos[13797]
with torch.no_grad():
    plane = model(normalise(np.asarray(ds.img[i])[None], dev)).argmax(1)[0].cpu().numpy().astype(np.uint8)
# plane: 0 = fundo, 1 = atleta A (gi azul), 2 = atleta B (gi branco), em 320x180
```

Para o visual, leia o frame em 720p do vídeo (`cap.set(cv2.CAP_PROP_POS_FRAMES, n)`)
e faça upscale do `plane` com `cv2.INTER_NEAREST` — é a resolução real de saída
do modelo, e a borda levemente serrilhada é honesta.

**Cores das máscaras** (mantenha, são as mesmas de todas as peças anteriores):
atleta A = vermelho `(0,0,255)` em BGR, atleta B = verde `(0,255,0)`.
Preencha com 45% de opacidade e contorne com 3px da cor cheia.

**Frames escolhidos** (já verificados: IoU alto, atletas entrelaçados, cena legível):

- hero: **13797** — gi branco por cima, azul por baixo, banners da IBJJF ao fundo
- thumbnails: **13347** (tartaruga), **17867** (cem quilos), **17951**, **13690**

Corte cada frame em 16:9 em volta da união das duas máscaras, com ~34% de folga,
para os atletas ficarem grandes. A luta é Galvão × Ribeiro, IBJJF Pro League
Grand Prix 2017.

## Identidade visual (obrigatória — já existe uma série)

Copie o estilo de `scripts/make_story_arch.py` e `scripts/make_story_student.py`,
que geram os cards de Stories do mesmo projeto. Reaproveite as funções se quiser.

```
fundo        #090D13      painel        #0F141F      painel destacado  #101A2E
azul         #3983F0      azul claro    #7FB0F7      régua             #1C2430
branco       #EFEFF0      cinza         #636F7E      apagado           #3E4A5C
```

Fontes: `/System/Library/Fonts/Supplemental/Arial Bold.ttf` e `Arial.ttf`.
Títulos em caixa alta. Um filete fino de 2px separando blocos. Zero gradiente,
zero sombra, zero ícone, zero emoji. Uma cor de destaque só. É linguagem de
figura de artigo científico, e é isso que dá a autoridade.

Estrutura fixa da série: sobrancelha `BJJVISION` em azul com tracking no topo;
rodapé com filete, `estudo de caso por` em cinza, `THIAGO ABREU` em branco
bold 52px, e uma barra azul de 340×6 embaixo do nome.

Margem lateral de 64px. Diferente do Story, o LinkedIn não tem UI cobrindo as
bordas, então pode usar de y=56 até y=1444.

## Números medidos — use estes, não outros

**O modelo**

- 3.350.339 parâmetros, 13,4 MB em disco, U-Net com 9 blocos convolucionais
- entrada 320×180×3, saída 3 classes (fundo, atleta A, atleta B)
- treinado do zero em 3.000 passos, **17 minutos no MacBook** (MPS)

**Parâmetros por bloco** (resolução · canais · parâmetros · % do total)

```
e1    320×180   32    10.208     0,3%
e2    160×90    64    55.552     1,7%
e3     80×45   128   221.696     6,6%
e4     40×23   256   885.760    26,4%
bott   20×12   256  1.180.672   35,2%
d4     40×23   128   737.792    22,0%
d3     80×45    64   184.576     5,5%
d2    160×90    32    46.208     1,4%
d1    320×180   32    27.776     0,8%
head  320×180    3        99     0,0%
```

e4 + bott + d4 = 84% de todo o peso.

**Validação** (4 shots inteiros deixados de fora do treino, 3.754 frames)

- IoU 0,798 com identidade atribuída · 0,864 ignorando identidade
- 0,0% de troca de identidade
- distribuição: p10 0,644 · mediana 0,825 · p90 0,918 · 83% dos frames acima de 0,70
- split **por shot**, nunca aleatório por frame: frames vizinhos são quase
  idênticos, e dentro de um shot a câmera, a luz e os dois quimonos são fixos

**Contra o professor** (SAM2 hiera large, que gerou os rótulos)

- SAM2: 224.446.898 parâmetros · 898 MB · 4,34 fps em GPU alugada
- aluno: 3.350.339 parâmetros · 13,4 MB · **251 fps no laptop**
- 67× menos parâmetros; uma luta de 20 mil frames sai em 1,3 min em vez de 77 min

**Resultados que dão credibilidade técnica** (use um ou dois, não todos)

- segurando o tamanho do atleta fixo, o IoU **sobe** com oclusão: 0,773 com
  0–40% escondido, 0,844 com 80–100%. O erro é escala, não oclusão.
- o aluno por frame é mais estável no tempo que o professor com memória:
  IoU entre frames consecutivos 0,895 contra 0,888

## Precisão — o que NÃO afirmar

- Não diga que dispensa GPU: o treino do professor gastou os 77 minutos alugados.
  O ganho é das próximas lutas. "Destilei" já comunica isso.
- Não diga que generaliza. Foi treinado nas máscaras de **uma** luta; o teste em
  outra luta ainda não foi feito.
- Os 0,0% de troca de identidade não provam que o modelo resolveu identidade:
  o atleta A é o gi azul nas três lutas disponíveis. Não venda isso como
  robustez.

## Entrega

1. Escreva `scripts/make_linkedin_student.py` com docstring explicando as
   decisões de layout, no mesmo tom dos outros scripts do repositório.
2. Gere `data/out/linkedin_student.png`.
3. **Abra a imagem gerada e olhe**, antes de dizer que terminou. Cheque:
   nada sobreposto, nada cortado na margem, acentos corretos (`parâmetros`,
   `máscaras`, `saída`), separador de milhar com ponto e decimal com vírgula.
   Se algo colidir, conserte e gere de novo.
4. Deixe os textos num dicionário no topo do script, para uma versão em inglês
   sair com uma flag.
