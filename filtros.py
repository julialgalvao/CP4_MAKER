"""Filtros visuais divertidos para a webcam da assistente Jack.

Todos os efeitos são desenhados apenas com OpenCV e NumPy, sem depender de
imagens externas. Cada filtro recebe o frame (imagem BGR) e a lista de rostos
detectados no formato (x, y, largura, altura) e devolve um novo frame já
processado.

O tema principal é a HUD estilo Homem de Ferro (inspirada na F.R.I.D.A.Y.),
mas também há filtros clássicos e bem-humorados (óculos escuros, cartoon,
térmico, etc.).
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _sobrepor_bgra(fundo, sobreposicao_bgr, mascara, x, y):
    """Cola `sobreposicao_bgr` em `fundo` usando `mascara` (0..255) na posição x,y.

    Recorta automaticamente o que ficar fora dos limites do frame.
    """
    altura_f, largura_f = fundo.shape[:2]
    altura_s, largura_s = sobreposicao_bgr.shape[:2]

    # Região visível dentro do frame.
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(largura_f, x + largura_s), min(altura_f, y + altura_s)
    if x1 >= x2 or y1 >= y2:
        return fundo

    # Região correspondente na sobreposição.
    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)

    recorte_fundo = fundo[y1:y2, x1:x2].astype(np.float32)
    recorte_sobre = sobreposicao_bgr[sy1:sy2, sx1:sx2].astype(np.float32)
    alfa = (mascara[sy1:sy2, sx1:sx2].astype(np.float32) / 255.0)[:, :, None]

    misturado = recorte_sobre * alfa + recorte_fundo * (1.0 - alfa)
    fundo[y1:y2, x1:x2] = misturado.astype(np.uint8)
    return fundo


# ---------------------------------------------------------------------------
# Filtros de imagem inteira
# ---------------------------------------------------------------------------
def filtro_nenhum(frame, faces, tempo=0):
    """Sem efeito: mostra a imagem original."""
    return frame


def filtro_cinza(frame, faces, tempo=0):
    """Preto e branco clássico."""
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(cinza, cv2.COLOR_GRAY2BGR)


def filtro_negativo(frame, faces, tempo=0):
    """Inverte as cores (efeito 'raio-x')."""
    return cv2.bitwise_not(frame)


def filtro_cartoon(frame, faces, tempo=0):
    """Deixa a imagem com aparência de desenho animado."""
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cinza = cv2.medianBlur(cinza, 5)
    bordas = cv2.adaptiveThreshold(
        cinza, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )
    cor = cv2.bilateralFilter(frame, 9, 250, 250)
    bordas_bgr = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(cor, bordas_bgr)


def filtro_termico(frame, faces, tempo=0):
    """Visão térmica colorida (mapa de cores JET)."""
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(cinza, cv2.COLORMAP_JET)


# ---------------------------------------------------------------------------
# Filtros que usam os rostos detectados
# ---------------------------------------------------------------------------
def filtro_oculos(frame, faces, tempo=0):
    """Desenha óculos escuros estilosos sobre cada rosto detectado."""
    for (x, y, largura, altura) in faces:
        # Os olhos ficam por volta de 38% da altura do rosto.
        olho_y = y + int(altura * 0.38)
        margem = int(largura * 0.10)
        esquerda = x + margem
        direita = x + largura - margem
        raio = max(8, int(largura * 0.16))
        centro_e = (esquerda + raio, olho_y)
        centro_d = (direita - raio, olho_y)

        # Lentes.
        cv2.circle(frame, centro_e, raio, (0, 0, 0), -1)
        cv2.circle(frame, centro_d, raio, (0, 0, 0), -1)
        # Brilho nas lentes.
        cv2.circle(frame, (centro_e[0] - raio // 3, olho_y - raio // 3),
                   max(2, raio // 5), (255, 255, 255), -1)
        cv2.circle(frame, (centro_d[0] - raio // 3, olho_y - raio // 3),
                   max(2, raio // 5), (255, 255, 255), -1)
        # Ponte e hastes.
        cv2.line(frame, (centro_e[0] + raio, olho_y),
                 (centro_d[0] - raio, olho_y), (0, 0, 0), 4)
        cv2.line(frame, (esquerda - raio, olho_y),
                 (x, olho_y - raio // 2), (0, 0, 0), 4)
        cv2.line(frame, (direita + raio, olho_y),
                 (x + largura, olho_y - raio // 2), (0, 0, 0), 4)
    return frame


def filtro_hud(frame, faces, tempo=0):
    """HUD estilo Homem de Ferro sobre cada rosto.

    Desenha uma mira animada com cantos, círculos girando e um texto de
    'alvo identificado'. `tempo` (contador de frames) controla a animação.
    """
    saida = frame.copy()
    ciano = (255, 255, 0)
    laranja = (0, 140, 255)

    for (x, y, largura, altura) in faces:
        centro = (x + largura // 2, y + altura // 2)
        raio = int(max(largura, altura) * 0.62)

        # Círculo externo pontilhado (feito com arcos girando).
        angulo = (tempo * 4) % 360
        for base in range(0, 360, 45):
            inicio = base + angulo
            cv2.ellipse(saida, centro, (raio, raio), 0, inicio, inicio + 25,
                        ciano, 2)

        # Círculo interno girando ao contrário.
        raio_interno = int(raio * 0.72)
        for base in range(0, 360, 90):
            inicio = base - angulo
            cv2.ellipse(saida, centro, (raio_interno, raio_interno), 0,
                        inicio, inicio + 40, laranja, 2)

        # Cantos da mira (colchetes) ao redor do rosto.
        c = int(largura * 0.25)
        cantos = [
            ((x, y), (x + c, y), (x, y + c)),
            ((x + largura, y), (x + largura - c, y), (x + largura, y + c)),
            ((x, y + altura), (x + c, y + altura), (x, y + altura - c)),
            ((x + largura, y + altura), (x + largura - c, y + altura),
             (x + largura, y + altura - c)),
        ]
        for canto, h, v in cantos:
            cv2.line(saida, canto, h, ciano, 2)
            cv2.line(saida, canto, v, ciano, 2)

        # Cruz central.
        cv2.line(saida, (centro[0] - 12, centro[1]),
                 (centro[0] + 12, centro[1]), ciano, 1)
        cv2.line(saida, (centro[0], centro[1] - 12),
                 (centro[0], centro[1] + 12), ciano, 1)

        # Rótulo do alvo.
        cv2.putText(saida, "ALVO IDENTIFICADO", (x, max(20, y - 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, ciano, 1, cv2.LINE_AA)

    # Moldura/HUD fixa nos cantos da tela.
    altura_f, largura_f = saida.shape[:2]
    marca = 40
    for (px, py, dx, dy) in [
        (10, 10, 1, 1),
        (largura_f - 10, 10, -1, 1),
        (10, altura_f - 10, 1, -1),
        (largura_f - 10, altura_f - 10, -1, -1),
    ]:
        cv2.line(saida, (px, py), (px + dx * marca, py), ciano, 2)
        cv2.line(saida, (px, py), (px, py + dy * marca), ciano, 2)

    cv2.putText(saida, "JACK // SISTEMA ONLINE", (18, altura_f - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ciano, 1, cv2.LINE_AA)
    return saida


# ---------------------------------------------------------------------------
# Registro de filtros — ordem usada pela tecla de troca.
# ---------------------------------------------------------------------------
FILTROS = [
    ("Normal", filtro_nenhum),
    ("HUD Homem de Ferro", filtro_hud),
    ("Oculos", filtro_oculos),
    ("Cartoon", filtro_cartoon),
    ("Termico", filtro_termico),
    ("Negativo", filtro_negativo),
    ("Preto e branco", filtro_cinza),
]


class GerenciadorFiltros:
    """Controla qual filtro está ativo e aplica no frame."""

    def __init__(self):
        self.indice = 0
        self.tempo = 0  # contador de frames para animações

    @property
    def nome_atual(self):
        return FILTROS[self.indice][0]

    def proximo(self):
        self.indice = (self.indice + 1) % len(FILTROS)

    def anterior(self):
        self.indice = (self.indice - 1) % len(FILTROS)

    def definir(self, indice):
        if 0 <= indice < len(FILTROS):
            self.indice = indice

    def aplicar(self, frame, faces):
        """Aplica o filtro atual e devolve o frame processado."""
        self.tempo += 1
        _, funcao = FILTROS[self.indice]
        return funcao(frame, faces, self.tempo)

    def selecionar_por_voz(self, texto):
        """Troca o filtro a partir de um comando de voz.

        Entende "proximo", "anterior", "normal" e também o nome de um filtro
        (ex.: "cartoon", "oculos", "termico", "homem de ferro", "negativo",
        "preto e branco"). Retorna o nome do filtro escolhido, ou None se
        não reconheceu nenhum comando.
        """
        if not texto:
            return None
        t = texto.lower()

        if "proxim" in t or "próxim" in t or "avanc" in t or "avanç" in t:
            self.proximo()
            return self.nome_atual
        if "anterior" in t or "volta" in t or "volte" in t:
            self.anterior()
            return self.nome_atual
        if "normal" in t or "nenhum" in t or "sem filtro" in t or "tira" in t:
            self.definir(0)
            return self.nome_atual

        # Palavras-chave que apontam para cada filtro.
        apelidos = {
            "homem de ferro": 1, "hud": 1, "ferro": 1, "iron": 1,
            "oculos": 2, "óculos": 2, "oculo": 2,
            "cartoon": 3, "desenho": 3,
            "termico": 4, "térmico": 4, "calor": 4,
            "negativo": 5, "raio x": 5, "raio-x": 5, "invert": 5,
            "preto e branco": 6, "cinza": 6, "preto": 6,
        }
        for palavra, indice in apelidos.items():
            if palavra in t:
                self.definir(indice)
                return self.nome_atual
        return None


def _carregar_detector():
    """Acha o Haar Cascade no projeto ou no OpenCV (a prova de versao)."""
    from pathlib import Path
    nome = "haarcascade_frontalface_default.xml"
    caminhos = [Path(__file__).parent / nome, Path(nome)]
    try:
        caminhos.append(Path(cv2.data.haarcascades) / nome)
    except Exception:
        pass
    for c in caminhos:
        if c.exists():
            d = cv2.CascadeClassifier(str(c))
            if not d.empty():
                return d
    return None


# Demonstração rápida sem depender do main: abre a webcam só com os filtros.
if __name__ == "__main__":
    detector = _carregar_detector()
    if detector is None:
        print("Nao encontrei haarcascade_frontalface_default.xml.")
        raise SystemExit
    camera = cv2.VideoCapture(0)
    gerenciador = GerenciadorFiltros()

    print("Teste de filtros. F troca o filtro, Q sai.")
    while camera.isOpened():
        ok, frame = camera.read()
        if not ok:
            break
        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(cinza, 1.2, 5, minSize=(80, 80))
        frame = gerenciador.aplicar(frame, faces)
        cv2.putText(frame, f"Filtro: {gerenciador.nome_atual}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Filtros Jack", frame)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla == ord("f"):
            gerenciador.proximo()

    camera.release()
    cv2.destroyAllWindows()
