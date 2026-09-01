# BjjVision — handoff para sessão Opus (2026-08-31, fim do dia)

Leia inteiro antes de agir. A memória automática do projeto carrega o resto;
`git log --oneline -25` conta o dia. **Não re-derive nada daqui: aja.**

## Por que esta sessão existe

A sessão anterior (Fable 5) estava construindo o conjunto-ouro de avaliação
julgando 200 cartões no olho — caro. O Thiago cortou: (1) o paulista23 é mau
exemplo (muita plateia, câmera longe, branco-vs-branco), (2) verificar a
hipótese "câmera muito aberta está fora do escopo do produto", (3) continuar
em Opus economizando créditos.

## Estado factual

- **v4 treinado** (`data/out/student_ckpt_v4/`): 0.7950 no held-out do galvao
  (melhor da história). Números held-out (0.18 paulista23 / 0.30 ferreira)
  são INVÁLIDOS: o gabarito do professor está contaminado nesses domínios
  (funde o par branco-branco, mascara staff/mesa/torcida; perde o atleta de
  azul na arena escura). O gargalo migrou do student para o professor.
- **Gold eval meio-construído** em `data/out/gold_eval/`:
  - 200 cartões (`card_*.jpg`, original|professor|student) + candidatos
    (`cand_*.npz` com planos teacher/student) + `samples.json`.
  - `verdicts.json`: 106/120 do paulista23 julgados → **1 ouro, 105 none**.
    Confirma o Thiago: mau exemplo. NÃO julgar os 14 restantes do paulista.
  - Ferreira (80 cartões, stacks 30–49) **não julgado**.
- **ANTHROPIC_API_KEY do ~/.zshrc está INVÁLIDA (401)** — o juiz VLM via API
  (`build_gold_eval.py judge`) não roda até o Thiago renovar. Julgar na
  sessão só com parcimônia (foi o que queimou créditos).
- Descoberta qualitativa já feita (registrada nos reasonings dos vereditos):
  no zoom fechado os DOIS candidatos acham os corpos (união boa; student
  união limpa, professor com vermelho em staffer) — o que falha é separar
  identidade branco-vs-branco. No plano aberto, tudo falha.
- Aluguel Vast: encerrado e destruído (~US$1). masks.bin das 5 lutas no
  laptop, verificados. **Backup externo do masks.bin continua pendente.**
- Export ONNX validado (`scripts/export_student.py`, paridade 1.8e-05).

## O que fazer, em ordem

1. **Hipótese da escala, SEM ler imagem cara**: os `cand_*.npz` já têm os
   planos por frame. Computar proxy de escala (altura do bbox da união dos
   candidatos / altura do frame) por cartão julgado e cruzar com os
   vereditos: taxa de "corpos achados" × escala. Complemento decisivo e
   barato: **experimento crop-zoom** — nos frames wide (calasans,
   paulista23), recortar 2× em volta da união do professor, redimensionar a
   320×180, rodar v4 no recorte, renderizar ~6 exemplos e OLHAR (uma leitura
   só, em pilha). Se a máscara ressuscitar: escala é a causa; "zoom digital"
   vira pré-processamento e o escopo do produto se define como "atletas ≥
   ~35–40% da altura do frame".
2. **Ouro do ferreira**: julgar os stacks 30–49 (80 cartões; se os créditos
   apertarem, metade alternada basta) — na sessão OU via API se a chave
   voltar. Critérios: cobre os dois atletas com máscaras próprias; nada em
   árbitro/torcida/LED/mesa; identidade ferreira: vermelho = kimono AZUL.
   Depois `python scripts/build_gold_eval.py assemble` e
   `python scripts/build_gold_eval.py eval --ckpt data/out/student_ckpt_v4/student.pt`.
3. **Registrar a decisão de escopo** no `config/fights.yaml` e na memória:
   paulista23 sai de held-out principal (vira `role: fora-de-escopo` ou
   similar, com o porquê). Propor ao Thiago o held-out regional CERTO:
   **vídeo filmado da beira do tatame** (ele pode filmar treino/campeonato
   próprio) — é o footage de deployment real; nada do YouTube cobre isso.
4. Se sobrar fôlego: o problema de pesquisa central é a **identidade
   branco-vs-branco** (âncora sem cor: textura/posição/propagação). A opção
   "tarefa-união" (atletas vs fundo, sem identidade) está madura — nos
   frames médios o student já entrega união limpa; um v5-união seria útil
   pro produto (dessaturar os dois, deixar o anotador dizer quem é quem).

## Regras da casa (custam caro quando esquecidas)

- Medir antes de afirmar; **números ordenam, olhos decidem** — provado nas
  duas direções (calasans 100%-falso; frag quase condenou 47 shots bons).
- Um commit por fato medido, mensagem com o porquê. Push após lote.
- Nunca rotular com saída de modelo sem gate independente.
- GPU alugada só depois do código rodar local. Smoke test antes de run cheio.
- Vereditos de juiz são versionados (`overrides.json`, `verdicts.json`).
