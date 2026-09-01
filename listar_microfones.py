"""Utilitário para listar os microfones e testar a captação de voz.

Use este script quando o Jack não estiver entendendo o que você fala.
Ele mostra o índice de cada microfone e faz um teste rápido de reconhecimento.

Como usar:
    python listar_microfones.py

Depois, se precisar, edite INDICE_MICROFONE no main.py com o número do
microfone que funcionou melhor (por exemplo, o microfone do MacBook).
"""
import speech_recognition as sr

try:
    import audioop  # medir volume (Python < 3.13)
except Exception:
    audioop = None


def listar():
    print("=" * 50)
    print("  MICROFONES DISPONIVEIS")
    print("=" * 50)
    for indice, nome in enumerate(sr.Microphone.list_microphone_names()):
        print(f"  [{indice}] {nome}")
    print("=" * 50)


def testar(indice=None):
    r = sr.Recognizer()
    onde = "microfone padrao" if indice is None else f"microfone [{indice}]"
    print(f"\nTestando o {onde}. Prepare-se para falar...")
    with sr.Microphone(device_index=indice) as mic:
        print("Ajustando ruido ambiente (1s)...")
        r.adjust_for_ambient_noise(mic, duration=1.0)
        print(f"Nivel de ruido (energy_threshold): {r.energy_threshold:.0f}")
        print(">>> FALE AGORA (algo como 'jack que horas sao') <<<")
        try:
            audio = r.listen(mic, timeout=6, phrase_time_limit=5)
        except sr.WaitTimeoutError:
            print("Nao detectei nenhuma fala. O microfone pode estar mudo/errado.")
            return

    if audioop is not None:
        rms = audioop.rms(audio.get_raw_data(), audio.sample_width)
        print(f"Volume captado (RMS): {rms}  (bom: acima de ~500)")

    print("Reconhecendo (Google, pt-BR)...")
    try:
        texto = r.recognize_google(audio, language="pt-BR")
        print(f"\n  >>> VOCE FALOU: {texto!r}\n")
        print("Microfone funcionando! Se este nao for o padrao, coloque o")
        print("indice dele em INDICE_MICROFONE no main.py.")
    except sr.UnknownValueError:
        print("\n  Nao consegui entender. Tente falar mais perto e mais alto,")
        print("  ou escolha outro microfone (veja a lista acima).")
    except Exception as erro:
        print(f"\n  Erro no reconhecimento: {type(erro).__name__}: {erro}")


def achar_indice(nome_parcial):
    for i, nome in enumerate(sr.Microphone.list_microphone_names()):
        if nome_parcial.lower() in (nome or "").lower():
            return i
    return None


if __name__ == "__main__":
    listar()
    entrada = input(
        "\nDigite o indice do microfone para testar\n"
        "(ENTER = testa o Fifine automaticamente): "
    ).strip()
    if entrada.isdigit():
        indice = int(entrada)
    else:
        indice = achar_indice("Fifine")
        if indice is None:
            print("Fifine nao encontrado; testando o microfone padrao.")
    testar(indice)
