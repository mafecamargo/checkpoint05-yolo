"""
CP5 - Applied Computer Vision

Integrantes: 
Fernanda Kaory Saito - RM551104
João Pedro Borsato da Cruz - RM550194
Maria Fernanda Vieira de Camargo - RM97956
Pedro Lucas de Andrade Nunes - RM550633
Vinicius Almeida Bernadino de Souza - RM97888
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import csv
import json
import urllib.request

import cv2
import numpy as np
import torch
from ultralytics import YOLO


DEVICE_YOLO = "mps" if torch.backends.mps.is_available() else "cpu"


# ============================================================
# LEITURA DO VIDEO
# ============================================================
def ler_video(caminho, lado_max=960):
    cap = cv2.VideoCapture(caminho)
    if not cap.isOpened():
        raise FileNotFoundError("Nao foi possivel abrir o video: %s" % caminho)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    quadros = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        escala = lado_max / max(h, w)
        if escala < 1:
            f = cv2.resize(f, (int(w * escala), int(h * escala)))
        quadros.append(f)
    cap.release()
    if not quadros:
        raise ValueError("O video nao tem quadros legiveis: %s" % caminho)
    return quadros, fps


# ============================================================
# 1) YOLO -> Detecção de pessoas
# ============================================================
class DetectorPessoa:
    def __init__(self, pesos="yolov8n.pt", conf=0.4):
        self.modelo = YOLO(pesos)
        self.conf = conf

    def __call__(self, quadro):
        r = self.modelo.predict(quadro, classes=[0], conf=self.conf,
                                device=DEVICE_YOLO, verbose=False)[0]
        if len(r.boxes) == 0:
            return None, 0.0
        i = int(r.boxes.conf.argmax())  
        caixa = r.boxes.xyxy[i].cpu().numpy().astype(int)
        return caixa, float(r.boxes.conf[i])


# ============================================================
# 2) UNet -> Segmentação de pessoas
# ============================================================
class SegmentadorDeepLab:
    """DeepLabV3+ pre-treinado."""
    CLASSE_PESSOA = 15

    def __init__(self):
        import keras_hub
        self.modelo = keras_hub.models.ImageSegmenter.from_preset(
            "deeplab_v3_plus_resnet50_pascalvoc")

    def __call__(self, recorte_bgr):
        h, w = recorte_bgr.shape[:2]
        rgb = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2RGB)
        esc = 512/ max(h, w)
        if esc < 1:
            rgb = cv2.resize(rgb, (round(w * esc), round(h * esc)))
        saida = self.modelo.predict(np.expand_dims(rgb, 0), verbose=0)
        classes = np.argmax(saida, axis=-1)[0]
        mascara = (classes == self.CLASSE_PESSOA).astype(np.uint8)
        mascara = cv2.resize(mascara, (w, h), interpolation=cv2.INTER_NEAREST)
        return mascara > 0


class SegmentadorUNet:
    """U-Net Keras: 256x256, pixels / 255, limiar 0.5."""

    def __init__(self, arquivo_modelo, tamanho=256, limiar=0.5):
        from keras.models import load_model
        self.modelo = load_model(arquivo_modelo)
        self.tamanho = tamanho
        self.limiar = limiar

    def __call__(self, recorte_bgr):
        h, w = recorte_bgr.shape[:2]
        img = cv2.resize(recorte_bgr, (self.tamanho, self.tamanho))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        prob = self.modelo.predict(np.expand_dims(img, 0), verbose=0)[0]
        if prob.ndim == 3:
            prob = prob[:, :, 0]
        mascara = (prob >= self.limiar).astype(np.uint8)
        mascara = cv2.resize(mascara, (w, h), interpolation=cv2.INTER_NEAREST)
        return mascara > 0


def mascara_no_quadro(quadro, caixa, segmentador, margem=0.10):
    """Recorta a caixa do YOLO (com margem), segmenta e devolve a máscara no quadro inteiro."""
    H, W = quadro.shape[:2]
    x1, y1, x2, y2 = caixa
    dx, dy = int((x2 - x1) * margem), int((y2 - y1) * margem)
    x1, y1 = max(0, x1 - dx), max(0, y1 - dy)
    x2, y2 = min(W, x2 + dx), min(H, y2 + dy)
    completa = np.zeros((H, W), dtype=bool)
    if x2 - x1 > 10 and y2 - y1 > 10:
        completa[y1:y2, x1:x2] = segmentador(quadro[y1:y2, x1:x2])
    return completa


# ============================================================
# 3) SlowFast -> Classificação de ação 
# ============================================================
NUMERO_QUADROS = 32
TAMANHO_IMAGEM = 224
ALPHA = 4
ARQUIVO_CLASSES = "kinetics_classnames.json"
URL_CLASSES = ("https://dl.fbaipublicfiles.com/pyslowfast/dataset/"
               "class_names/kinetics_classnames.json")


class ClassificadorAcao:
    def __init__(self):
        self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        self.modelo = torch.hub.load("facebookresearch/pytorchvideo",
                                     "slowfast_r50", pretrained=True)
        self.modelo = self.modelo.to(self.dispositivo).eval()
        if not os.path.exists(ARQUIVO_CLASSES):
            urllib.request.urlretrieve(URL_CLASSES, ARQUIVO_CLASSES)
        with open(ARQUIVO_CLASSES) as f:
            dados = json.load(f)
        self.classes = {int(v): k.strip('"') for k, v in dados.items()}

    @staticmethod
    def preparar(quadros_bgr):
        idx = np.linspace(0, len(quadros_bgr) - 1, NUMERO_QUADROS).astype(int)
        clip = []
        for i in idx:
            f = cv2.cvtColor(quadros_bgr[i], cv2.COLOR_BGR2RGB)
            h, w = f.shape[:2]
            esc = TAMANHO_IMAGEM / min(h, w)
            f = cv2.resize(f, (max(TAMANHO_IMAGEM, round(w * esc)),
                               max(TAMANHO_IMAGEM, round(h * esc))))
            h, w = f.shape[:2]
            y, x = (h - TAMANHO_IMAGEM) // 2, (w - TAMANHO_IMAGEM) // 2
            clip.append(f[y:y + TAMANHO_IMAGEM, x:x + TAMANHO_IMAGEM])
        video = torch.from_numpy(np.stack(clip)).float() / 255.0  # T,H,W,C
        video = (video - 0.45) / 0.225                            # normalizacao
        video = video.permute(3, 0, 1, 2)                          # C,T,H,W
        idx_lento = torch.linspace(0, NUMERO_QUADROS - 1, NUMERO_QUADROS // ALPHA).long()
        caminho_lento = video.index_select(1, idx_lento)
        return [caminho_lento.unsqueeze(0), video.unsqueeze(0)]   # [slow, fast]

    def __call__(self, quadros_bgr, k=3):
        caminhos = [c.to(self.dispositivo) for c in self.preparar(quadros_bgr)]
        with torch.no_grad():
            logits = self.modelo(caminhos)
            probabilidades = torch.softmax(logits, dim=1)[0]
        top = probabilidades.topk(k)
        return [(self.classes[int(i)], float(p)) for p, i in zip(top.values, top.indices)]


def recortar_janela(clip, caixas):
    """Extensao opcional: SlowFast so na regiao da pessoa (uniao das caixas)."""
    validas = [c for c in caixas if c is not None]
    if not validas:
        return clip
    c = np.array(validas)
    x1, y1 = c[:, 0].min(), c[:, 1].min()
    x2, y2 = c[:, 2].max(), c[:, 3].max()
    return [f[y1:y2, x1:x2] for f in clip]


# ============================================================
# 4) VISUALIZACAO
# ============================================================
def desenhar(quadro, caixa, conf, mascara, acao):
    out = quadro.copy()
    if mascara is not None and mascara.any():
        verde = out.copy()
        verde[mascara] = (0, 255, 0)
        out = cv2.addWeighted(out, 0.6, verde, 0.4, 0)
    if caixa is not None:
        x1, y1, x2, y2 = map(int, caixa)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out, "pessoa %.2f" % conf, (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    if acao is not None:
        nome, prob = acao
        cv2.rectangle(out, (0, 0), (out.shape[1], 32), (0, 0, 0), -1)
        cv2.putText(out, "Acao: %s (%.0f%%)" % (nome, prob * 100), (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out

def fechar_janelas():
    cv2.destroyAllWindows()
    cv2.waitKey(1)  

# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="saida")
    ap.add_argument("--seg", choices=["deeplab", "unet"], default="deeplab")
    ap.add_argument("--modelo-unet", default="modelo_unet.keras")
    ap.add_argument("--janela", type=float, default=2.0, help="segundos por janela do SlowFast")
    ap.add_argument("--crop", action="store_true", help="SlowFast apenas no recorte da pessoa")
    ap.add_argument("--sem-janela", action="store_true", help="nao mostrar o video na tela")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mostrar = not a.sem_janela

    quadros, fps = ler_video(a.video)
    print("%d quadros @ %.1f fps | YOLO em %s" % (len(quadros), fps, DEVICE_YOLO))

    print("Carregando modelos...")
    detector = DetectorPessoa()
    segmentador = (SegmentadorUNet(a.modelo_unet) if a.seg == "unet"
                   else SegmentadorDeepLab())
    classificador = ClassificadorAcao()

    # Etapas 1 e 2: YOLO + segmentacao quadro a quadro (com visualização ao vivo)
    caixas, confs, mascaras = [], [], []
    for n, q in enumerate(quadros):
        caixa, conf = detector(q)
        caixas.append(caixa)
        confs.append(conf)
        mascaras.append(mascara_no_quadro(q, caixa, segmentador) if caixa is not None else None)
        if n % 30 == 0:
            print("  quadro %d/%d" % (n, len(quadros)))
        if mostrar:
            img = desenhar(q, caixa, conf, mascaras[-1], None)
            cv2.imshow("CP5 - YOLO + segmentação quadro a quadro", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                mostrar = False
                break
    if mostrar:
        fechar_janelas()

    # Etapa 3: SlowFast em janelas consecutivas
    tam = max(8, int(round(a.janela * fps)))
    print("Classificando ações com SlowFast")
    acao_quadro = [None] * len(quadros)
    linhas = []
    for ini in range(0, len(quadros), tam):
        fim = min(ini + tam, len(quadros))
        if fim - ini < 8:
            break
        clip = quadros[ini:fim]
        if a.crop:
            clip = recortar_janela(clip, caixas[ini:fim])
        top = classificador(clip)
        for i in range(ini, fim):
            acao_quadro[i] = top[0]
        linha = ["%.2f" % (ini / fps), "%.2f" % (fim / fps)]
        for nome, p in top:
            linha += [nome, "%.3f" % p]
        linhas.append(linha)
        print("[%5.1fs - %5.1fs] %s (%.0f%%)" % (ini / fps, fim / fps, top[0][0], top[0][1] * 100))

    # Logs para o relatorio
    with open(os.path.join(a.out, "acoes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["inicio_s", "fim_s", "top1", "conf1", "top2", "conf2", "top3", "conf3"])
        w.writerows(linhas)
    with open(os.path.join(a.out, "quadros.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quadro", "tempo_s", "pessoa_detectada", "conf_yolo", "area_mascara_pct"])
        for i, m in enumerate(mascaras):
            area = 100.0 * m.mean() if m is not None else 0.0
            w.writerow([i, "%.2f" % (i / fps), caixas[i] is not None,
                        "%.3f" % confs[i], "%.2f" % area])

    # Vídeo de demonstração
    H, W = quadros[0].shape[:2]
    gravador = cv2.VideoWriter(os.path.join(a.out, "demo.mp4"),
                               cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    momentos = {int(len(quadros) * p) for p in (0.25, 0.5, 0.75)}
    mostrar = not a.sem_janela
    espera = max(1, int(1000 / fps * 0.5))  
    for i, q in enumerate(quadros):
        img = desenhar(q, caixas[i], confs[i], mascaras[i], acao_quadro[i])
        gravador.write(img)
        if i in momentos:
            cv2.imwrite(os.path.join(a.out, "momento_%05d.png" % i), img)
        if mostrar:
            cv2.imshow("CP5 - Demonstração (Q = fechar)", img)
            if cv2.waitKey(espera) & 0xFF == ord("q"):
                mostrar = False
                break
    gravador.release()
    fechar_janelas()
    print("Pronto! Resultados em:", a.out)


if __name__ == "__main__":
    main()
