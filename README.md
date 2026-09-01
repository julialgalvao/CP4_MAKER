# 🤖 Projeto Jack — Assistente Virtual (Checkpoint 4)

Assistente virtual em Python controlada por **voz**. A assistente se chama **Jack** e só executa comandos depois de ouvir seu nome (wake word). Ela reconhece fala, responde por voz, gerencia uma agenda, faz cálculos, reconhece rostos pela webcam (com **filtros visuais divertidos**), conversa com uma **IA generativa** e ainda tem vários comandos extras.

> Projeto desenvolvido para o Checkpoint 4 — 2º semestre.

## 👥 Integrantes do grupo

| Nome | RM |
| --- | --- |
| Aline Fernandes Zeppelini | RM97966 |
| Camilly Breitbach Ishida | RM551474 |
| Julia Leite Galvão | RM550201 |

---

## ✅ Checklist de requisitos (itens 1 a 10 do enunciado)

| # | Requisito do enunciado | Status | Onde está no código |
| --- | --- | :---: | --- |
| 1 | Wake word: só age depois de ouvir o nome **"Jack"**; sem o nome, apenas imprime o texto | ✅ | `iniciar()` + `contem_nome_assistente()` em `main.py` |
| 2 | Cadastrar evento e salvar em `agenda.txt` | ✅ | `cadastrar_evento()` |
| 3 | Ler agenda e falar os eventos | ✅ | `ler_agenda()` |
| 4 | Informar a hora atual | ✅ | `informar_hora()` |
| 5 | Informar a data atual | ✅ | `informar_data()` |
| 6 | Calcular (+, −, ×, ÷) | ✅ | `calcular()` |
| 7 | Reconhecer face / "Quem sou eu?" pela webcam | ✅ | `reconhecer_face()` (EigenFaces) |
| 8 | Limpar agenda **sem** apagar o arquivo | ✅ | `limpar_agenda()` |
| 9 | Integração com IA generativa (Groq na nuvem ou Ollama local) | ✅ | `perguntar_ia()` |
| 10 | Comandos extras (mínimo 2 — **fizemos vários**) | ✅ | ver seção abaixo |

**Critérios de avaliação atendidos:**

- ✅ **[3 pts]** Itens 1 a 9 completos, código dividido em funções/métodos e comentado.
- ✅ **[1 pt]** Código orientado a objetos com pelo menos uma classe (`class Jack`).
- ✅ **[3 pts]** Muito mais que 2 comandos extras (item 10): cotações em tempo real, filtros de imagem, cadastro de novos rostos e apresentação.
- ✅ **[2 pts]** Repositório público com documentação, dependências e instruções (este README).
- ✅ **[1 pt]** Base pronta para uma apresentação clara e objetiva.

---

## ✨ Funcionalidades

### Comandos obrigatórios

- **Wake word "Jack"** — nenhuma ação acontece sem o nome ser falado. O reconhecedor aceita variações comuns de transcrição (jaque, check, etc.) para não falhar na apresentação.
- **Agenda** — cadastrar, ler e limpar eventos (`agenda.txt`).
- **Hora e data atuais**.
- **Calculadora** por voz (adição, subtração, multiplicação e divisão).
- **Reconhecimento facial** pela webcam usando EigenFaces (OpenCV).
- **IA generativa** — Groq (nuvem) com fallback automático para Ollama (local).

### Comandos extras (item 10)

1. 🔎 **Pesquisar no Google**
2. ▶️ **Abrir vídeo no YouTube**
3. 🖼️ **Tirar print da tela**
4. 🎓 **Abrir o portal da FIAP**
5. 💵 **Cotação do dólar** em tempo real (AwesomeAPI, gratuita)
6. ₿ **Cotação do bitcoin** em tempo real
7. 🎭 **Filtros visuais divertidos** na webcam (troca por tecla ou por voz)
8. 🧑‍💻 **Cadastrar novos rostos por voz** (captura fotos e treina o modelo na hora)
9. 🙋 **Apresentação automática** — o Jack se apresenta ao iniciar

> A fala usa **voz feminina em português** (Luciana, via `say` no macOS; `pyttsx3` no Windows).

### 🎭 Filtros visuais da webcam

Durante o "Quem sou eu?", dá para trocar o visual da câmera ao vivo. O reconhecimento facial continua funcionando por baixo de qualquer filtro.

| Atalho | Ação |
| :---: | --- |
| **F** | Próximo filtro |
| **N** | Filtro anterior |
| **0** | Voltar ao normal |
| **Q** | Fechar a webcam |

Filtros disponíveis: `Normal`, `HUD` (mira animada), `Óculos` escuros, `Cartoon`, `Térmico`, `Negativo` e `Preto e branco`.

> Para testar só os filtros, sem a assistente inteira: `python filtros.py`

---

## 📁 Estrutura do projeto

```text
projeto_friday_cp4/
├── main.py                             # Assistente Jack e todos os comandos (classe Jack)
├── filtros.py                          # Filtros visuais divertidos da webcam
├── capturar_faces.py                   # Coleta fotos pela webcam (script à parte)
├── treinar_faces.py                    # Treina o modelo EigenFaces e gera o .yml
├── listar_microfones.py                # Utilitário para testar/escolher o microfone
├── haarcascade_frontalface_default.xml # Detector de rostos (incluso no projeto)
├── agenda.txt                          # Eventos cadastrados
├── pessoas.txt                         # IDs e nomes usados no reconhecimento
├── fotos/                              # Fotos de treinamento (já incluídas)
├── classificadoreigen.yml              # Modelo facial pré-treinado
├── prints/                             # Prints da tela gerados pelo comando extra
├── requirements.txt                    # Dependências
├── .env.example                        # Exemplo de variável da Groq
└── .gitignore
```

---

## 🛠️ Instalação

Recomendado: **Python 3.11 ou 3.12**.

### 1. Criar o ambiente virtual

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

Se o **PyAudio** falhar na instalação, atualize o pip primeiro:

```bash
python -m pip install --upgrade pip
pip install PyAudio
```

> No macOS pode ser necessário `brew install portaudio` antes do PyAudio.
> No Linux, `sudo apt install portaudio19-dev`.

---

## 🧠 Configurar a IA generativa

Há duas opções. Basta uma delas funcionar.

### Opção A — Groq (nuvem, gratuita)

1. Crie uma chave em https://console.groq.com/.
2. Defina a variável de ambiente antes de rodar:

**Windows (PowerShell):**

```powershell
$env:GROQ_API_KEY="SUA_CHAVE"
python main.py
```

**macOS / Linux:**

```bash
export GROQ_API_KEY="SUA_CHAVE"
python main.py
```

O modelo usado é `openai/gpt-oss-120b`.

### Opção B — Ollama (local, funciona offline)

```bash
# macOS (Homebrew)
brew install ollama
brew services start ollama      # deixa o Ollama rodando em segundo plano
ollama pull llama3.2:3b         # baixa o modelo (~2 GB)
```

Com o Ollama rodando, execute `python main.py`. Se a `GROQ_API_KEY` não estiver configurada, o Jack usa o Ollama automaticamente — sem internet e sem chave.

> ⚠️ Nunca suba sua chave da Groq para o GitHub. Use `.env` (que está no `.gitignore`).

---

## 🍎 Permissões no macOS

Na primeira execução, o macOS pede permissões. Autorize o app do Terminal em **Ajustes do Sistema → Privacidade e Segurança**:

- **Microfone** — para o Jack ouvir os comandos
- **Câmera** — para o reconhecimento facial
- **Gravação de Tela** — para o comando "tirar print" (depois de ligar, **feche e reabra o Terminal**)

---

## 📷 Reconhecimento facial

O reconhecimento usa `cv2.face.EigenFaceRecognizer_create()` (pacote **opencv-contrib-python**, já no `requirements.txt`). O detector de rostos (`haarcascade_frontalface_default.xml`) e um modelo pré-treinado (`classificadoreigen.yml`) já vêm no projeto.

Quem estiver cadastrado aparece com o nome (retângulo **verde**); quem não estiver aparece como **"Desconhecido"** (retângulo **vermelho**).

### Cadastrar novas pessoas — pela voz (recomendado)

Fale **"Jack cadastrar rosto"**. O Jack pergunta o nome, abre a câmera, você aperta **ESPAÇO** para tirar as fotos e ele **treina o modelo sozinho** no final.

### Cadastrar por script (alternativa)

```bash
python capturar_faces.py   # tira as fotos
python treinar_faces.py    # treina e gera o classificadoreigen.yml
```

> O valor `LIMITE_CONFIANCA_FACE` em `main.py` pode ser ajustado conforme a iluminação. No EigenFaces, valores **menores** indicam maior proximidade com o rosto treinado.

---

## ▶️ Como executar

```bash
python main.py
```

Ao iniciar, o Jack mostra um banner com os integrantes e **se apresenta sozinho** (fala quem é e o que faz). Depois fica ouvindo o microfone.

> **Importante:** todo comando precisa começar com **"Jack"**. Sem o nome, o Jack apenas mostra o texto na tela e não faz nada (regra do enunciado). Para encerrar: **"Jack sair"**.

---

## 🎙️ Comandos (fale sempre começando com "Jack")

```text
Jack que horas são
Jack que dia é hoje
Jack quanto é 10 mais 5
Jack cadastrar evento na agenda
Jack ler agenda
Jack limpar agenda
Jack quem sou eu
Jack cadastrar rosto
Jack explique o que é inteligência artificial
Jack quem é você
Jack pesquisar no Google inteligência artificial
Jack abrir YouTube tutorial de Python
Jack tirar um print da tela
Jack abrir portal da faculdade
Jack qual o valor do dólar hoje
Jack quanto vale um bitcoin
Jack sair
```

- Sem o nome **"Jack"**, o comando é ignorado (regra do enunciado).
- A calculadora aceita **mais, menos, vezes, dividido por**.
- Se não entender, o Jack pede para repetir.

## 📷 Na câmera ("Jack quem sou eu")

- **F** próximo filtro · **N** anterior · **0** normal · **ESPAÇO** trocar filtro por voz · **Q** fechar
- **"Jack cadastrar rosto"**: ele pergunta o nome, abre a câmera, você aperta **ESPAÇO** para tirar as fotos e ele treina sozinho.

---

## 📦 Dependências

Veja `requirements.txt`:

- `SpeechRecognition` — reconhecimento de voz
- `PyAudio` — acesso ao microfone
- `pyttsx3` — síntese de voz (fala)
- `opencv-contrib-python` — webcam, detecção e EigenFaces
- `numpy` — processamento das imagens
- `requests` — IA local (Ollama) e cotações
- `groq` — IA generativa na nuvem
- `PyAutoGUI` — print da tela (no macOS o Jack usa o `screencapture` nativo)

---

## 🚀 Subir no GitHub

```bash
git init
git add .
git commit -m "Checkpoint 4 - Assistente Jack"
git branch -M main
git remote add origin URL_DO_SEU_REPOSITORIO
git push -u origin main
```
