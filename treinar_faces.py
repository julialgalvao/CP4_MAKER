from pathlib import Path

import cv2
import numpy as np

PASTA_FOTOS = Path("fotos")
MODELO_FACE = Path("classificadoreigen.yml")


def carregar_imagens():
    faces = []
    codigos = []

    for caminho in sorted(PASTA_FOTOS.glob("*.jpg")):
        partes = caminho.stem.split("_", 2)
        if len(partes) < 3 or not partes[0].isdigit():
            continue

        codigo = int(partes[0])
        imagem = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)
        if imagem is None:
            continue

        imagem = cv2.resize(imagem, (220, 220))
        faces.append(imagem)
        codigos.append(codigo)

    return np.array(codigos), faces


def main():
    if not hasattr(cv2, "face"):
        print("Instale opencv-contrib-python para usar EigenFaces.")
        return

    codigos, faces = carregar_imagens()

    if len(faces) < 2:
        print("Não há fotos suficientes. Execute capturar_faces.py primeiro.")
        return

    reconhecedor = cv2.face.EigenFaceRecognizer_create()
    reconhecedor.train(faces, codigos)
    reconhecedor.write(str(MODELO_FACE))

    print(f"Modelo treinado com {len(faces)} imagens e salvo em {MODELO_FACE}.")


if __name__ == "__main__":
    main()
