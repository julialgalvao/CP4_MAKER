import re
from pathlib import Path

import cv2
import numpy as np

PASTA_FOTOS = Path("fotos")
ARQUIVO_PESSOAS = Path("pessoas.txt")
QUANTIDADE_FOTOS = 25
TAMANHO = (220, 220)
ARQUIVO_CASCADE = "haarcascade_frontalface_default.xml"


def carregar_detector_faces():
    """Procura o Haar Cascade no projeto e, se não achar, no OpenCV."""
    caminhos = [Path(__file__).parent / ARQUIVO_CASCADE, Path(ARQUIVO_CASCADE)]
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


def carregar_pessoas():
    pessoas = {}
    if ARQUIVO_PESSOAS.exists():
        for linha in ARQUIVO_PESSOAS.read_text(encoding="utf-8").splitlines():
            if ";" in linha:
                codigo, nome = linha.split(";", 1)
                pessoas[int(codigo)] = nome
    return pessoas


def salvar_pessoas(pessoas):
    linhas = [f"{codigo};{nome}" for codigo, nome in sorted(pessoas.items())]
    ARQUIVO_PESSOAS.write_text("\n".join(linhas) + "\n", encoding="utf-8")


def normalizar_nome(nome):
    nome = re.sub(r"[^a-zA-ZÀ-ÿ0-9_-]", "_", nome.strip())
    return nome or "pessoa"


def main():
    PASTA_FOTOS.mkdir(exist_ok=True)
    pessoas = carregar_pessoas()

    nome = input("Digite o nome da pessoa: ").strip()
    if not nome:
        print("Nome inválido.")
        return

    codigo = max(pessoas.keys(), default=0) + 1
    pessoas[codigo] = nome
    salvar_pessoas(pessoas)

    detector = carregar_detector_faces()
    if detector is None:
        print("Não encontrei haarcascade_frontalface_default.xml na pasta do projeto.")
        return
    camera = cv2.VideoCapture(0)
    amostra = 1

    print("Pressione F para capturar uma foto quando o rosto estiver enquadrado.")

    while camera.isOpened() and amostra <= QUANTIDADE_FOTOS:
        status, imagem = camera.read()
        if not status:
            break

        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            imagem_cinza, scaleFactor=1.05, minNeighbors=5, minSize=(120, 120)
        )

        for x, y, largura, altura in faces:
            cv2.rectangle(imagem, (x, y), (x + largura, y + altura), (0, 0, 255), 2)

            if cv2.waitKey(1) & 0xFF == ord("f"):
                # Mesma ideia do material de referência: evita fotos muito escuras.
                if np.average(imagem_cinza) > 80:
                    rosto = cv2.resize(
                        imagem_cinza[y:y + altura, x:x + largura], TAMANHO
                    )
                    arquivo = PASTA_FOTOS / f"{codigo}_{normalizar_nome(nome)}_{amostra}.jpg"
                    cv2.imwrite(str(arquivo), rosto)
                    print(f"Foto {amostra}/{QUANTIDADE_FOTOS} salva: {arquivo}")
                    amostra += 1
                else:
                    print("Imagem muito escura. Tente novamente.")

        cv2.putText(
            imagem,
            f"Fotos: {amostra - 1}/{QUANTIDADE_FOTOS} - F captura / Q sai",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cv2.imshow("Captura de rostos", imagem)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    print("Captura finalizada.")


if __name__ == "__main__":
    main()
