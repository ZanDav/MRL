import os
import subprocess
import shutil

# --- Configurazione ---
# Nome della cartella (dentro A) che contiene le sottocartelle con i video originali
INPUT_SUBFOLDER_NAME = "MineRLTreechop-v0" # CAMBIA QUESTO SE NECESSARIO

# Nome della cartella (dentro A) dove verranno salvati i video convertiti
OUTPUT_SUBFOLDER_NAME = "VideoConvertiti_640x360"

TARGET_WIDTH = 640
TARGET_HEIGHT = 360
PAD_COLOR = "black" # Colore per il padding (letterbox/pillarbox)
# --- Fine Configurazione ---

def check_ffmpeg():
    """Verifica se FFmpeg è installato e nel PATH."""
    if shutil.which("ffmpeg") is None:
        print("ERRORE: FFmpeg non trovato. Assicurati che sia installato e aggiunto al PATH di sistema.")
        print("Puoi scaricarlo da: https://ffmpeg.org/download.html")
        return False
    return True

def convert_video(input_path, output_path):
    """
    Converte un video alla risoluzione target mantenendo l'aspect ratio
    originale e aggiungendo padding se necessario.
    """
    ffmpeg_command = [
        "ffmpeg",
        "-i", input_path,
        "-vf", f"scale=w={TARGET_WIDTH}:h={TARGET_HEIGHT}:force_original_aspect_ratio=decrease,pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color={PAD_COLOR}",
        "-c:a", "copy",
        "-y",
        output_path
    ]

    # print(f"  Comando FFmpeg: {' '.join(ffmpeg_command)}") # Decommenta per debug dettagliato

    try:
        process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print(f"  SUCCESS: Convertito '{os.path.basename(input_path)}' in '{os.path.basename(output_path)}'")
            return True
        else:
            print(f"  ERRORE durante la conversione di '{os.path.basename(input_path)}':")
            # Decodifica l'output di ffmpeg, ignorando errori se ci sono caratteri non standard
            print(f"    FFmpeg stdout: {stdout.decode(errors='ignore')}")
            print(f"    FFmpeg stderr: {stderr.decode(errors='ignore')}")
            return False
    except FileNotFoundError:
        print(f"  ERRORE: FFmpeg non trovato durante l'esecuzione del comando per '{os.path.basename(input_path)}'.")
        return False
    except Exception as e:
        print(f"  ERRORE SCONOSCIUTO durante la conversione di '{os.path.basename(input_path)}': {e}")
        return False

def main():
    if not check_ffmpeg():
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_base_dir = os.path.join(script_dir, INPUT_SUBFOLDER_NAME)
    output_base_dir = os.path.join(script_dir, OUTPUT_SUBFOLDER_NAME)

    if not os.path.isdir(input_base_dir):
        print(f"ERRORE: La cartella di input '{INPUT_SUBFOLDER_NAME}' non esiste in '{script_dir}'.")
        print("Controlla la configurazione 'INPUT_SUBFOLDER_NAME'.")
        return

    os.makedirs(output_base_dir, exist_ok=True)
    print(f"Cartella di output: '{output_base_dir}' (tutti i video verranno salvati qui direttamente)")

    converted_count = 0
    failed_count = 0

    for root, _, files in os.walk(input_base_dir):
        for filename in files:
            if filename.lower().endswith(".mp4"):
                input_video_path = os.path.join(root, filename)
                print(f"\nTrovato video: {input_video_path}")

                # Genera un nome file di output "appiattito" basato sul percorso relativo
                # Questo evita collisioni se file con lo stesso nome esistono in sottocartelle diverse
                # Esempio: se input_video_path è "INPUT_SUBFOLDER_NAME/subdir/video.mp4",
                #          relative_path sarà "subdir/video.mp4" (o "subdir\video.mp4" su Windows)
                #          e output_filename diventerà "subdir_video.mp4".
                # Se il video è nella radice di INPUT_SUBFOLDER_NAME, il nome non cambia.
                relative_path_from_input_root = os.path.relpath(input_video_path, input_base_dir)
                
                # Sostituisci i separatori di directory (es. / o \) con un underscore
                output_filename = relative_path_from_input_root.replace(os.sep, "_")
                
                # Il percorso di output sarà direttamente nella cartella output_base_dir
                output_video_path = os.path.join(output_base_dir, output_filename)

                # Non è più necessario creare sottocartelle di output qui,
                # la cartella output_base_dir è già stata creata.

                if convert_video(input_video_path, output_video_path):
                    converted_count += 1
                else:
                    failed_count += 1

    print(f"\n--- Riepilogo ---")
    print(f"Video convertiti con successo: {converted_count}")
    print(f"Conversioni fallite: {failed_count}")
    print(f"I video convertiti sono in: '{output_base_dir}'")

if __name__ == "__main__":
    main()