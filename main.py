import os
import re
import subprocess
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import cv2
import numpy as np
import pyautogui
import pyttsx3
import requests
import speech_recognition as sr

# Módulo próprio com os filtros visuais divertidos da webcam.
import filtros

try:
    from groq import Groq
except ImportError:
    Groq = None


# ---------------------------- CONFIGURAÇÕES ----------------------------
NOME_ASSISTENTE = "jack"
# Variações que o reconhecimento de voz costuma transcrever para "jack".
# Isso evita que o comando seja ignorado só por causa da grafia do Google.
VARIACOES_NOME = ("jack", "jaque", "jeck", "jak", "check", "jack")
IDIOMA = "pt-BR"
ARQUIVO_AGENDA = Path("agenda.txt")
MODELO_FACE = Path("classificadoreigen.yml")
ARQUIVO_PESSOAS = Path("pessoas.txt")
PASTA_FOTOS = Path("fotos")          # onde ficam as fotos de treino
TAMANHO_ROSTO = (220, 220)           # tamanho padrão dos rostos
QUANTIDADE_FOTOS = 25                # fotos por pessoa no cadastro
# EigenFaces: quanto MENOR a confiança, mais parecido com o rosto treinado.
# Abaixo deste valor consideramos "reconhecido"; acima, "Desconhecido".
# Ajuste conforme a iluminação/câmera (valores típicos ao vivo: 3000 a 8000).
LIMITE_CONFIANCA_FACE = 8000

# Valor especial devolvido por ouvir() quando captou som mas não entendeu.
NAO_ENTENDI = "__nao_entendi__"

# Microfone a usar. Em vez de um índice fixo (que muda quando um aparelho
# conecta/desconecta), procuramos pelo NOME. Tentamos os nomes desta lista,
# na ordem, e usamos o primeiro que existir. Se nenhum existir, usa o padrão.
# Deixe "" na lista para permitir cair no microfone padrão do sistema.
MICROFONES_PREFERIDOS = ["Fifine", "MacBook Pro Microphone"]

# Índice resolvido em tempo de execução (não mexa aqui).
INDICE_MICROFONE = None

# Nome do arquivo do detector de rostos (Haar Cascade).
ARQUIVO_CASCADE = "haarcascade_frontalface_default.xml"


def carregar_detector_faces():
    """Cria o detector de rostos procurando o cascade em vários lugares.

    Algumas versões novas do OpenCV não trazem o arquivo em cv2.data, então
    priorizamos uma cópia local no projeto. Assim funciona em qualquer máquina.
    """
    caminhos = [
        Path(__file__).parent / ARQUIVO_CASCADE,   # cópia dentro do projeto
        Path(ARQUIVO_CASCADE),                       # pasta atual
    ]
    # Caminho embutido no OpenCV (quando existir).
    try:
        caminhos.append(Path(cv2.data.haarcascades) / ARQUIVO_CASCADE)
    except Exception:
        pass

    for caminho in caminhos:
        if caminho.exists():
            detector = cv2.CascadeClassifier(str(caminho))
            if not detector.empty():
                return detector

    return None


class Jack:
    """Assistente virtual do projeto CP4."""

    def __init__(self):
        self.reconhecedor = sr.Recognizer()
        # Deixa o reconhecedor mais tolerante e estável:
        # - não recalibra o ruído sozinho a cada escuta (evita "surdez");
        # - espera um pouco mais de silêncio antes de considerar o fim da fala.
        self.reconhecedor.dynamic_energy_threshold = False
        self.reconhecedor.energy_threshold = 300
        self.reconhecedor.pause_threshold = 0.9
        self.sintetizador = pyttsx3.init()
        self.sintetizador.setProperty("volume", 1.0)
        self.sintetizador.setProperty("rate", 190)
        self._selecionar_voz_portugues()
        # Resolve qual microfone usar (por nome) uma única vez.
        global INDICE_MICROFONE
        INDICE_MICROFONE = self._resolver_microfone()
        self.ativa = False
        ARQUIVO_AGENDA.touch(exist_ok=True)

    @staticmethod
    def _resolver_microfone():
        """Descobre o índice do microfone preferido pelo NOME.

        Assim, se um aparelho conectar/desconectar (mudando os índices),
        continuamos pegando o microfone certo. Retorna None (padrão) se
        nenhum dos preferidos estiver disponível.
        """
        try:
            nomes = sr.Microphone.list_microphone_names()
        except Exception:
            return None

        print("Microfones detectados:")
        for i, nome in enumerate(nomes):
            print(f"  [{i}] {nome}")

        for preferido in MICROFONES_PREFERIDOS:
            if not preferido:
                continue
            for i, nome in enumerate(nomes):
                if preferido.lower() in (nome or "").lower():
                    print(f">> Usando microfone: [{i}] {nome}\n")
                    return i

        print(">> Nenhum microfone preferido encontrado; usando o padrão.\n")
        return None

    def _selecionar_voz_portugues(self):
        """Escolhe uma voz FEMININA em português para a fala ficar natural.

        Preferimos a Luciana (voz feminina PT-BR do macOS). Se não existir,
        cai para qualquer voz feminina em português e, por último, qualquer
        voz em português. Funciona em qualquer sistema sem quebrar.
        """
        self._voz_id = None
        try:
            vozes = self.sintetizador.getProperty("voices")

            def eh_portugues(voz):
                idiomas = " ".join(
                    str(item) for item in (getattr(voz, "languages", []) or [])
                ).lower()
                return "pt" in idiomas or "portug" in (voz.name or "").lower()

            def eh_feminina(voz):
                genero = str(getattr(voz, "gender", "")).lower()
                nome = (voz.name or "").lower()
                # Nomes tipicamente femininos disponíveis em PT-BR.
                nomes_fem = ("luciana", "joana", "flo", "sandy", "shelley", "grandma")
                return "female" in genero or any(n in nome for n in nomes_fem)

            def usar(voz_id):
                # Guarda o id para reforçar a voz em cada fala.
                self._voz_id = voz_id
                self.sintetizador.setProperty("voice", voz_id)

            # 0) Tenta o id conhecido da Luciana diretamente (macOS).
            id_luciana = "com.apple.voice.compact.pt-BR.Luciana"
            if any(v.id == id_luciana for v in vozes):
                usar(id_luciana)
                return

            # 1) Luciana pelo nome (caso o id seja diferente).
            for voz in vozes:
                if "luciana" in (voz.name or "").lower():
                    usar(voz.id)
                    return
            # 2) Qualquer voz feminina em português.
            for voz in vozes:
                if eh_portugues(voz) and eh_feminina(voz):
                    usar(voz.id)
                    return
            # 3) Qualquer voz em português.
            for voz in vozes:
                if eh_portugues(voz):
                    usar(voz.id)
                    return
        except Exception:
            # Se algo der errado, seguimos com a voz padrão do sistema.
            pass

    # ---------------------------- VOZ ----------------------------
    def falar(self, mensagem):
        """Mostra a resposta na tela e fala em voz alta (voz feminina).

        No macOS usamos o comando nativo `say` (mais confiável e com a voz
        Luciana). Em outros sistemas, usamos o pyttsx3.
        """
        print(f"Jack: {mensagem}")

        # 1) macOS: comando nativo `say` (voz feminina Luciana).
        if sys.platform == "darwin":
            try:
                subprocess.run(
                    ["say", "-v", "Luciana", "-r", "195", mensagem],
                    check=True,
                )
                return
            except Exception as erro:
                print(f"(say falhou, usando pyttsx3: {erro})")

        # 2) Outros sistemas (ou fallback): pyttsx3.
        if getattr(self, "_voz_id", None):
            try:
                self.sintetizador.setProperty("voice", self._voz_id)
            except Exception:
                pass
        self.sintetizador.say(mensagem)
        self.sintetizador.runAndWait()

    def ouvir(self):
        """Escuta o microfone e transforma a fala em texto.

        Retorna:
        - o texto reconhecido (str), quando entende a fala;
        - "" quando há silêncio ou erro de microfone/serviço;
        - NAO_ENTENDI quando captou som mas não conseguiu transcrever.
        """
        try:
            with sr.Microphone(device_index=INDICE_MICROFONE) as mic:
                # Calibra o ruído só uma vez (na primeira escuta), para não
                # ficar "surdo" recalibrando toda hora com o próprio silêncio.
                if not getattr(self, "_ruido_calibrado", False):
                    print("Calibrando ruído do ambiente (fique em silêncio)...")
                    self.reconhecedor.adjust_for_ambient_noise(mic, duration=1.0)
                    self._ruido_calibrado = True
                print("Ouvindo...")
                audio = self.reconhecedor.listen(mic, timeout=8, phrase_time_limit=10)

            print("Processando áudio...")
            texto = self.reconhecedor.recognize_google(audio, language=IDIOMA)
            print(f"Você falou: {texto}")
            return texto.strip()

        except sr.WaitTimeoutError:
            print("Nenhuma fala foi detectada.")
        except sr.UnknownValueError:
            # Houve som, mas o serviço não conseguiu transcrever.
            print("Não consegui entender o áudio.")
            return NAO_ENTENDI
        except sr.RequestError as erro:
            print(f"Erro no serviço de reconhecimento: {erro}")
        except OSError as erro:
            print(f"Erro ao acessar o microfone: {erro}")

        return ""

    # ---------------------------- AGENDA ----------------------------
    def cadastrar_evento(self):
        self.falar("Ok, qual evento devo cadastrar?")
        evento = self.ouvir()

        if not evento or evento == NAO_ENTENDI:
            self.falar("Não consegui entender o evento. Tente cadastrar de novo.")
            return

        agora = datetime.now().strftime("%d/%m/%Y %H:%M")
        with ARQUIVO_AGENDA.open("a", encoding="utf-8") as arquivo:
            arquivo.write(f"[{agora}] {evento}\n")

        self.falar("Evento cadastrado na agenda.")

    def ler_agenda(self):
        conteudo = ARQUIVO_AGENDA.read_text(encoding="utf-8").strip()

        if not conteudo:
            self.falar("A agenda está vazia.")
            return

        print("\n--- AGENDA ---")
        print(conteudo)
        print("--------------\n")
        self.falar("Vou ler os eventos cadastrados.")

        for linha in conteudo.splitlines():
            # Retira a data do começo para a fala ficar mais natural.
            evento = re.sub(r"^\[[^\]]+\]\s*", "", linha)
            self.falar(evento)

    def limpar_agenda(self):
        # Abre em modo de escrita para apagar o conteúdo sem excluir o arquivo.
        ARQUIVO_AGENDA.write_text("", encoding="utf-8")
        self.falar("Agenda limpa com sucesso.")

    # ---------------------------- DATA E HORA ----------------------------
    def informar_hora(self):
        agora = datetime.now().strftime("%H:%M")
        self.falar(f"Agora são {agora}.")

    def informar_data(self):
        hoje = datetime.now()
        meses = [
            "janeiro", "fevereiro", "março", "abril", "maio", "junho",
            "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"
        ]
        data = f"{hoje.day} de {meses[hoje.month - 1]} de {hoje.year}"
        self.falar(f"Hoje é {data}.")

    # ---------------------------- CALCULADORA ----------------------------
    def calcular(self, comando):
        """Calcula adição, subtração, multiplicação e divisão.

        Aceita tanto "calcular 10 mais 5" quanto frases naturais como
        "quanto é 2 mais 2", "quanto que é 2 + 2", "qual o resultado de...".
        """
        expressao = comando.lower()

        # Remove as várias formas de pedir uma conta, deixando só a expressão.
        prefixos = [
            "calcular", "calcula", "calcule",
            "quanto que é", "quanto que e", "quanto é", "quanto e",
            "quanto da", "quanto dá", "qual é o resultado de",
            "qual e o resultado de", "qual o resultado de", "resultado de",
            "quanto vale",
        ]
        for prefixo in prefixos:
            expressao = re.sub(rf"\b{re.escape(prefixo)}\b", "", expressao)

        # Remove palavras/gírias que não fazem parte da conta.
        for lixo in ("baby", "por favor", "pra mim", "para mim", "hein", "então"):
            expressao = expressao.replace(lixo, "")

        expressao = expressao.replace(",", ".").strip(" ?!.")

        substituicoes = {
            "dividido por": "/",
            "dividido": "/",
            "dividido por": "/",
            "multiplicado por": "*",
            "multiplicado": "*",
            "vezes": "*",
            "mais": "+",
            "menos": "-",
            "somar": "+",
            "soma": "+",
            "com": "+",
            " e ": "+",
            "x": "*",
        }

        for palavra, simbolo in substituicoes.items():
            expressao = expressao.replace(palavra, simbolo)

        # Remove espaços para facilitar o casamento do padrão.
        expressao = expressao.replace(" ", "")

        padrao = r"^(-?\d+(?:\.\d+)?)([+\-*/])(-?\d+(?:\.\d+)?)$"
        resultado_regex = re.match(padrao, expressao)

        if not resultado_regex:
            self.falar("Não entendi a conta. Fale algo como: quanto é 10 mais 5.")
            return

        numero1 = float(resultado_regex.group(1))
        operador = resultado_regex.group(2)
        numero2 = float(resultado_regex.group(3))

        if operador == "+":
            resultado = numero1 + numero2
        elif operador == "-":
            resultado = numero1 - numero2
        elif operador == "*":
            resultado = numero1 * numero2
        else:
            if numero2 == 0:
                self.falar("Não é possível dividir por zero.")
                return
            resultado = numero1 / numero2

        if resultado.is_integer():
            resultado = int(resultado)

        self.falar(f"O resultado é {resultado}.")

    # ---------------------------- RECONHECIMENTO FACIAL ----------------------------
    @staticmethod
    def carregar_pessoas():
        pessoas = {}
        if not ARQUIVO_PESSOAS.exists():
            return pessoas

        for linha in ARQUIVO_PESSOAS.read_text(encoding="utf-8").splitlines():
            if ";" in linha:
                codigo, nome = linha.split(";", 1)
                pessoas[int(codigo)] = nome
        return pessoas

    @staticmethod
    def _salvar_pessoas(pessoas):
        linhas = [f"{cod};{nome}" for cod, nome in sorted(pessoas.items())]
        ARQUIVO_PESSOAS.write_text("\n".join(linhas) + "\n", encoding="utf-8")

    @staticmethod
    def _normalizar_nome(nome):
        nome = re.sub(r"[^a-zA-ZÀ-ÿ0-9_-]", "_", nome.strip())
        return nome or "pessoa"

    def treinar_modelo(self):
        """Treina o EigenFaces com todas as fotos e salva o .yml."""
        faces, codigos = [], []
        for caminho in sorted(PASTA_FOTOS.glob("*.jpg")):
            partes = caminho.stem.split("_", 2)
            if len(partes) < 3 or not partes[0].isdigit():
                continue
            imagem = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)
            if imagem is None:
                continue
            faces.append(cv2.resize(imagem, TAMANHO_ROSTO))
            codigos.append(int(partes[0]))

        if len(faces) < 2:
            self.falar("Não há fotos suficientes para treinar o modelo.")
            return False

        reconhecedor = cv2.face.EigenFaceRecognizer_create()
        reconhecedor.train(faces, np.array(codigos))
        reconhecedor.write(str(MODELO_FACE))
        print(f"Modelo treinado com {len(faces)} imagens.")
        return True

    def cadastrar_rosto(self):
        """Cadastra um novo rosto: pergunta o nome, abre a câmera, captura
        as fotos (tecla ESPAÇO) e treina o modelo ao final."""
        if not hasattr(cv2, "face"):
            self.falar("O OpenCV instalado não tem o módulo face. Instale opencv-contrib-python.")
            return

        detector = carregar_detector_faces()
        if detector is None:
            self.falar("Não encontrei o detector de rostos.")
            return

        self.falar("Qual é o nome da pessoa que vou cadastrar?")
        nome = self.ouvir()
        if not nome or nome == NAO_ENTENDI:
            self.falar("Não entendi o nome. Cancelando o cadastro.")
            return

        PASTA_FOTOS.mkdir(exist_ok=True)
        pessoas = self.carregar_pessoas()
        codigo = max(pessoas.keys(), default=0) + 1
        pessoas[codigo] = nome
        self._salvar_pessoas(pessoas)

        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            self.falar("Não consegui abrir a webcam.")
            return

        self.falar(
            f"Vou cadastrar {nome}. Olhe para a câmera e aperte espaço para "
            f"tirar cada foto. Preciso de {QUANTIDADE_FOTOS} fotos. Aperte Q para cancelar."
        )

        amostra = 1
        while camera.isOpened() and amostra <= QUANTIDADE_FOTOS:
            status, imagem = camera.read()
            if not status:
                break

            cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
            # Detecção mais tolerante no cadastro (facilita achar o rosto).
            faces = detector.detectMultiScale(
                cinza, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )

            for x, y, largura, altura in faces:
                cv2.rectangle(imagem, (x, y), (x + largura, y + altura), (0, 255, 0), 2)

            tem_rosto = len(faces) > 0
            aviso = "ESPACO=foto" if tem_rosto else "POSICIONE O ROSTO"
            cor = (0, 255, 0) if tem_rosto else (0, 165, 255)
            cv2.putText(
                imagem,
                f"{nome}: {amostra - 1}/{QUANTIDADE_FOTOS}  {aviso}  Q=sair",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2,
            )
            cv2.imshow("Cadastro de rosto - Jack", imagem)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            if tecla == ord(" "):
                if tem_rosto:
                    # Salva o maior rosto detectado (mais próximo da câmera).
                    x, y, largura, altura = max(faces, key=lambda f: f[2] * f[3])
                    rosto = cv2.resize(cinza[y:y + altura, x:x + largura], TAMANHO_ROSTO)
                    arquivo = PASTA_FOTOS / f"{codigo}_{self._normalizar_nome(nome)}_{amostra}.jpg"
                    cv2.imwrite(str(arquivo), rosto)
                    print(f"Foto {amostra}/{QUANTIDADE_FOTOS} salva: {arquivo}")
                    amostra += 1
                else:
                    print("Nenhum rosto detectado no momento; ajuste a posição.")

        camera.release()
        cv2.destroyAllWindows()

        capturadas = amostra - 1
        if capturadas < 2:
            self.falar(
                "Não consegui tirar fotos suficientes. Verifique se o rosto "
                "aparece na câmera e tente cadastrar de novo."
            )
            return

        self.falar(f"Capturei {capturadas} fotos de {nome}. Agora vou treinar o modelo.")
        if self.treinar_modelo():
            self.falar(f"Pronto! Agora eu já consigo reconhecer {nome}.")

    def reconhecer_face(self):
        """Abre a webcam e reconhece a pessoa pelo modelo EigenFaces treinado."""
        if not MODELO_FACE.exists():
            self.falar("O modelo facial ainda não foi treinado. Execute capturar_faces.py e treinar_faces.py.")
            return

        if not hasattr(cv2, "face"):
            self.falar("O OpenCV instalado não possui o módulo face. Instale opencv-contrib-python.")
            return

        pessoas = self.carregar_pessoas()
        detector = carregar_detector_faces()
        if detector is None:
            self.falar(
                "Não encontrei o detector de rostos. Verifique o arquivo "
                "haarcascade_frontalface_default.xml na pasta do projeto."
            )
            return
        reconhecedor_face = cv2.face.EigenFaceRecognizer_create()
        reconhecedor_face.read(str(MODELO_FACE))
        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            self.falar("Não consegui abrir a webcam.")
            return

        # Gerenciador dos filtros visuais divertidos.
        gerenciador_filtros = filtros.GerenciadorFiltros()

        self.falar(
            "Abrindo a câmera. Aperte F para trocar o filtro, espaço para trocar "
            "por voz, e Q para fechar."
        )

        nome_ja_falado = None  # evita repetir a fala do nome a cada frame

        while True:
            status, imagem = camera.read()
            if not status:
                break

            # O reconhecimento sempre usa a imagem original em tons de cinza,
            # independentemente do filtro escolhido para a exibição.
            imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
            # minNeighbors mais alto e rosto maior evitam deteccoes falsas
            # (como a regiao do pescoco) que apareciam antes.
            faces = detector.detectMultiScale(
                imagem_cinza, scaleFactor=1.15, minNeighbors=8, minSize=(110, 110)
            )

            # Aplica o filtro visual escolhido pelo usuário sobre uma cópia.
            imagem = gerenciador_filtros.aplicar(imagem, faces)

            nome_principal = None
            for x, y, largura, altura in faces:
                rosto = cv2.resize(
                    imagem_cinza[y:y + altura, x:x + largura], (220, 220)
                )
                codigo, confianca = reconhecedor_face.predict(rosto)

                if confianca <= LIMITE_CONFIANCA_FACE and codigo in pessoas:
                    nome = pessoas[codigo]
                    cor = (0, 255, 0)  # verde = reconhecido
                else:
                    nome = "Desconhecido"
                    cor = (0, 0, 255)  # vermelho = desconhecido

                if nome != "Desconhecido" and nome_principal is None:
                    nome_principal = nome

                cv2.rectangle(imagem, (x, y), (x + largura, y + altura), cor, 2)
                cv2.putText(
                    imagem,
                    f"{nome} ({confianca:.0f})",
                    (x, max(30, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    cor,
                    2,
                )

            # Fala o nome da pessoa reconhecida (apenas uma vez por pessoa).
            if nome_principal and nome_principal != nome_ja_falado:
                self.falar(f"Reconheci você! Você é a {nome_principal}.")
                nome_ja_falado = nome_principal

            # Legenda com o filtro atual e os atalhos disponíveis.
            cv2.putText(
                imagem,
                f"Filtro: {gerenciador_filtros.nome_atual}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                imagem,
                "F: proximo  N: anterior  0: normal  ESPACO: voz  Q: sair",
                (10, imagem.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Reconhecimento facial - Jack", imagem)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            elif tecla == ord("f"):
                gerenciador_filtros.proximo()
            elif tecla == ord("n"):
                gerenciador_filtros.anterior()
            elif tecla == ord("0"):
                gerenciador_filtros.definir(0)
            elif tecla == ord(" "):
                # Troca o filtro por voz: escuta um comando e aplica.
                self.falar("Qual filtro você quer?")
                comando_voz = self.ouvir()
                if comando_voz and comando_voz != NAO_ENTENDI:
                    novo = gerenciador_filtros.selecionar_por_voz(comando_voz)
                    if novo:
                        self.falar(f"Filtro {novo}.")
                    else:
                        self.falar("Não reconheci esse filtro.")
                else:
                    self.falar("Não entendi. Tente de novo.")

        camera.release()
        cv2.destroyAllWindows()

    # ---------------------------- APRESENTAÇÃO ----------------------------
    def apresentar(self):
        """Resposta curta em que o Jack se apresenta."""
        self.falar("Olá! Eu sou o Jack, o assistente virtual do grupo.")
        self.falar(
            "Eu funciono por voz e só ajo quando você fala o meu nome antes do "
            "comando."
        )
        self.falar(
            "Eu gerencio sua agenda, faço contas, reconheço rostos na webcam e "
            "converso usando inteligência artificial, entre outras coisas."
        )
        self.falar("É só falar Jack seguido do comando que você quer.")

    # ---------------------------- IA GENERATIVA ----------------------------
    def perguntar_ia(self, pergunta):
        """Usa Groq quando há chave configurada; caso contrário tenta Ollama local."""
        pergunta = pergunta.strip()
        if not pergunta:
            self.falar("O que você quer perguntar para a IA?")
            pergunta = self.ouvir()

        if pergunta == NAO_ENTENDI:
            self.falar("Não entendi sua pergunta. Pode repetir?")
            return

        if not pergunta:
            return

        chave_groq = os.getenv("GROQ_API_KEY", "").strip()

        if chave_groq and Groq is not None:
            try:
                client = Groq(api_key=chave_groq)
                resposta = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {
                            "role": "system",
                            "content": "Você é a Jack, uma assistente útil. Responda em português e de forma breve, em no máximo 60 palavras.",
                        },
                        {"role": "user", "content": pergunta},
                    ],
                )
                texto = resposta.choices[0].message.content
                self.falar(texto)
                return
            except Exception as erro:
                print(f"Falha ao consultar Groq: {erro}")

        # Fallback local, semelhante ao exemplo disponibilizado em aula.
        try:
            resposta = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:3b",
                    "system": (
                        "Você é a Jack, uma assistente de voz. Responda SEMPRE "
                        "em português do Brasil, de forma curta e direta, no "
                        "máximo 2 frases. NUNCA use código, listas ou markdown, "
                        "pois sua resposta será lida em voz alta."
                    ),
                    "prompt": pergunta,
                    "stream": False,
                    "options": {"temperature": 0.5, "num_predict": 120},
                },
                timeout=45,
            )
            resposta.raise_for_status()
            texto = resposta.json().get("response", "Não recebi uma resposta da IA.")
            # Limpa eventuais blocos de código/markdown que atrapalham a fala.
            texto = re.sub(r"```.*?```", "", texto, flags=re.DOTALL)
            texto = texto.replace("`", "").replace("*", "").strip()
            if not texto:
                texto = "Não consegui gerar uma resposta agora."
            self.falar(texto)
        except Exception:
            self.falar(
                "Não consegui acessar a IA. Configure GROQ_API_KEY ou deixe o Ollama rodando com o modelo llama3.2:3b."
            )

    # ---------------------------- COMANDOS EXTRAS ----------------------------
    def pesquisar_google(self, termo):
        if not termo:
            self.falar("O que devo pesquisar no Google?")
            termo = self.ouvir()
        if termo and termo != NAO_ENTENDI:
            webbrowser.open(f"https://www.google.com/search?q={quote_plus(termo)}")
            self.falar(f"Pesquisando {termo} no Google.")
        else:
            self.falar("Não entendi o que pesquisar. Tente de novo.")

    def abrir_youtube(self, termo):
        if not termo:
            self.falar("Qual vídeo você quer procurar?")
            termo = self.ouvir()
        if termo and termo != NAO_ENTENDI:
            webbrowser.open(f"https://www.youtube.com/results?search_query={quote_plus(termo)}")
            self.falar(f"Abrindo resultados sobre {termo} no YouTube.")
        else:
            self.falar("Não entendi qual vídeo procurar. Tente de novo.")

    def tirar_print(self):
        pasta = Path("prints")
        pasta.mkdir(exist_ok=True)
        nome_arquivo = pasta / f"print_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        # No macOS usamos o comando nativo `screencapture` (mais confiável e
        # não depende do pyscreeze/Pillow). Em outros sistemas, PyAutoGUI.
        try:
            if sys.platform == "darwin":
                subprocess.run(["screencapture", "-x", str(nome_arquivo)], check=True)
            else:
                pyautogui.screenshot(str(nome_arquivo))
            self.falar("Pronto, tirei um print da tela.")
        except Exception as erro:
            print(f"Falha ao tirar print: {erro}")
            self.falar("Não consegui tirar o print da tela.")

    def abrir_portal_fiap(self):
        webbrowser.open("https://on.fiap.com.br/")
        self.falar("Abrindo o portal da FIAP.")

    def consultar_cotacao(self, par, nome_amigavel):
        """Consulta uma cotação na AwesomeAPI (gratuita, sem chave).

        `par` é algo como "USD-BRL" ou "BTC-BRL".
        """
        try:
            resposta = requests.get(
                f"https://economia.awesomeapi.com.br/last/{par}", timeout=15
            )
            resposta.raise_for_status()
            dados = resposta.json()
            chave = par.replace("-", "")
            valor = float(dados[chave]["bid"])
            # Formata em reais no padrão brasileiro (1.234,56).
            valor_formatado = f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            self.falar(f"O valor de {nome_amigavel} hoje é R$ {valor_formatado}.")
        except Exception as erro:
            print(f"Falha ao consultar cotação: {erro}")
            self.falar(f"Não consegui consultar a cotação de {nome_amigavel} agora.")

    def cotacao_dolar(self):
        self.consultar_cotacao("USD-BRL", "um dólar")

    def cotacao_bitcoin(self):
        self.consultar_cotacao("BTC-BRL", "um bitcoin")

    # ---------------------------- PROCESSAMENTO ----------------------------
    @staticmethod
    def contem_nome_assistente(texto):
        """Verifica se o texto contém o nome da assistente (ou uma variação)."""
        texto = texto.lower()
        return any(re.search(rf"\b{re.escape(nome)}\b", texto) for nome in VARIACOES_NOME)

    @staticmethod
    def remover_nome_assistente(texto):
        """Remove a primeira ocorrência do nome (ou variação) do texto."""
        for nome in VARIACOES_NOME:
            padrao = rf"\b{re.escape(nome)}\b"
            if re.search(padrao, texto, flags=re.IGNORECASE):
                return re.sub(padrao, "", texto, count=1, flags=re.IGNORECASE).strip(" ,.!?")
        return texto.strip(" ,.!?")

    def executar_comando(self, comando):
        comando_original = comando.strip()
        comando = comando_original.lower().strip()

        if not comando:
            self.falar("Como posso ajudar?")
            proximo = self.ouvir()
            if proximo == NAO_ENTENDI:
                self.falar("Desculpe, não entendi o que você disse. Pode repetir?")
            elif proximo:
                self.executar_comando(proximo)
            return

        # --- AGENDA --- (ler/limpar primeiro; cadastrar cobre várias frases)
        if "ler agenda" in comando or "ler a agenda" in comando or "ler minha agenda" in comando:
            self.ler_agenda()
        elif "limpar agenda" in comando or "limpar a agenda" in comando or "apagar agenda" in comando or "esvaziar agenda" in comando:
            self.limpar_agenda()
        elif (
            "agenda" in comando
            or "evento" in comando
            or "compromisso" in comando
            or "no meu calendário" in comando
            or "no calendário" in comando
        ):
            # Qualquer pedido relacionado a agenda/evento cai no cadastro,
            # ex.: "cadastrar um evento", "criar um evento", "marcar na agenda".
            self.cadastrar_evento()
        elif "que horas" in comando or "horas são" in comando:
            self.informar_hora()
        elif "que dia" in comando or "dia é hoje" in comando:
            self.informar_data()
        elif (
            comando.startswith("calcular")
            or comando.startswith("calcula")
            or comando.startswith("calcule")
            or "quanto é" in comando
            or "quanto e " in comando
            or "quanto que é" in comando
            or "quanto que e" in comando
            or "quanto da" in comando
            or "quanto dá" in comando
            or "resultado de" in comando
        ):
            self.calcular(comando_original)
        elif (
            "cadastrar rosto" in comando
            or "cadastrar novo rosto" in comando
            or "cadastrar face" in comando
            or "cadastrar uma pessoa" in comando
            or "cadastrar pessoa" in comando
            or "capturar rosto" in comando
            or "novo rosto" in comando
            or "adicionar rosto" in comando
        ):
            self.cadastrar_rosto()
        elif "reconhecer face" in comando or "quem sou eu" in comando or "reconhecer rosto" in comando:
            self.reconhecer_face()
        elif comando.startswith("perguntar ia"):
            pergunta = re.sub(r"^perguntar\s+ia", "", comando_original, flags=re.IGNORECASE).strip()
            self.perguntar_ia(pergunta)
        elif comando.startswith("perguntar para ia") or comando.startswith("perguntar para a ia"):
            pergunta = re.sub(r"^perguntar\s+para\s+(a\s+)?ia", "", comando_original, flags=re.IGNORECASE).strip()
            self.perguntar_ia(pergunta)
        elif "google" in comando or "pesquisar" in comando or "pesquise" in comando:
            # Extrai o termo depois de "google" (ou de "pesquisar/pesquise").
            termo = re.split(
                r"(?:no\s+)?google|pesquisar|pesquise",
                comando_original, flags=re.IGNORECASE,
            )[-1]
            termo = re.sub(r"^\s*(no\s+google|por)?\s*", "", termo, flags=re.IGNORECASE).strip(" ?.!")
            self.pesquisar_google(termo)
        elif "youtube" in comando or "you tube" in comando:
            # Extrai o termo depois de "youtube", ignorando conectivos comuns.
            termo = re.split(r"you\s*tube", comando_original, flags=re.IGNORECASE, maxsplit=1)[-1]
            # Remove conectivos apenas quando são palavras inteiras (\b),
            # para não comer a primeira letra de palavras como "opencv".
            termo = re.sub(
                r"^\s*(e\s+)?(procure|pesquise|rode\s+um\s+vídeo\s+sobre|um\s+vídeo\s+sobre|sobre)\b\s*",
                "", termo, flags=re.IGNORECASE,
            ).strip(" ?.!")
            self.abrir_youtube(termo)
        elif "print" in comando or "captura de tela" in comando or "foto da tela" in comando or "printar" in comando:
            self.tirar_print()
        elif "portal" in comando or "faculdade" in comando or "fiap" in comando:
            self.abrir_portal_fiap()
        elif "dólar" in comando or "dolar" in comando:
            self.cotacao_dolar()
        elif "bitcoin" in comando or "bit coin" in comando:
            self.cotacao_bitcoin()
        elif (
            "quem é você" in comando
            or "quem e voce" in comando
            or "quem é vc" in comando
            or "o que você faz" in comando
            or "o que voce faz" in comando
            or "o que você pode fazer" in comando
            or "o que voce pode fazer" in comando
            or "o que sabe fazer" in comando
            or "se apresente" in comando
            or "apresente-se" in comando
            or "quem é o jack" in comando
        ):
            self.apresentar()
        elif comando in {"sair", "encerrar", "desligar"}:
            self.falar("Até mais!")
            raise SystemExit
        else:
            # Comando válido após o wake word, mas não pertence à lista fixa:
            # usa a IA generativa como fallback.
            self.perguntar_ia(comando_original)

    @staticmethod
    def _mostrar_banner():
        """Imprime um cabeçalho organizado no terminal para a apresentação."""
        print("\n" + "=" * 60)
        print("        JACK - ASSISTENTE VIRTUAL  (Checkpoint 4)")
        print("=" * 60)
        print("  Integrantes:")
        print("   - Aline Fernandes Zeppelini  (RM97966)")
        print("   - Camilly Breitbach Ishida   (RM551474)")
        print("   - Julia Leite Galvao         (RM550201)")
        print("-" * 60)
        print("  Fale sempre 'Jack' antes do comando. Exemplos:")
        print("   * Jack que horas sao        * Jack que dia e hoje")
        print("   * Jack calcular 10 mais 5   * Jack cadastrar evento na agenda")
        print("   * Jack ler agenda           * Jack limpar agenda")
        print("   * Jack quem sou eu          * Jack cadastrar rosto")
        print("   * Jack qual o valor do dolar hoje")
        print("   * Jack quanto vale um bitcoin")
        print("   * Jack pesquisar no Google ...   * Jack abrir YouTube ...")
        print("   * Jack tirar um print da tela    * Jack abrir portal da faculdade")
        print("   * Jack explique o que e inteligencia artificial")
        print("   * Jack quem e voce   |   Para sair: Jack sair")
        print("=" * 60 + "\n")

    def iniciar(self):
        # Banner no terminal, deixando tudo pronto para a apresentação.
        self._mostrar_banner()
        # Ao iniciar, o Jack já se apresenta sozinho (sem precisar de comando)
        # e mostra alguns exemplos do que sabe fazer.
        self.apresentar()
        print(f"\n[ Ouvindo... diga '{NOME_ASSISTENTE}' antes do comando ]\n")

        while True:
            texto = self.ouvir()

            # Captou som mas não entendeu: pede para repetir (fallback falado).
            if texto == NAO_ENTENDI:
                self.falar("Desculpe, não entendi o que você disse. Pode repetir, por favor?")
                continue

            # Silêncio ou erro de microfone: apenas volta a ouvir.
            if not texto:
                continue

            # Regra principal do trabalho: sem falar o nome da assistente,
            # o texto é apenas mostrado na tela e nenhuma ação é executada.
            if not self.contem_nome_assistente(texto):
                print(f"Comando ignorado (sem wake word): {texto}")
                continue

            comando = self.remover_nome_assistente(texto)

            # Se só falou o nome, sem comando: mostra que entendeu e ajuda.
            if not comando:
                self.falar("Oi! Estou aqui. O que você gostaria que eu fizesse?")
                continue

            self.executar_comando(comando)


if __name__ == "__main__":
    assistente = Jack()
    assistente.iniciar()
