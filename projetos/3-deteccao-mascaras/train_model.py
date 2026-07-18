import os
import shutil

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Projeto 3 — Detecção de Máscaras Faciais (Fine-tuning do YOLO11n)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo pré-treinado YOLO11n: YOLO("yolo11n.pt")
#      (única exceção à regra de "sem modelos pré-treinados" do processo seletivo)
#   2. Fazer fine-tuning em dataset/data.yaml, em CPU (device="cpu"),
#      com um número de épocas modesto (ex: 15-30)
#   3. Copiar os pesos resultantes (results.save_dir / "weights" / "best.pt")
#      para "model.pt", na raiz desta pasta
# ---------------------------------------------------------------------------

# insira seu código aqui

# Dica de estrutura (não é obrigatório seguir exatamente assim):
#
# model = YOLO("yolo11n.pt")
# results = model.train(
#     data="dataset/data.yaml",
#     epochs=...,
#     imgsz=...,
#     batch=...,
#     device="cpu",
# )
# shutil.copy(results.save_dir / "weights" / "best.pt", "model.pt")


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_yaml = os.path.join(
        project_dir,
        "dataset",
        "data.yaml"
    )

    print("Carregando YOLO11n...")
    model = YOLO("yolo11n.pt")

    print("Iniciando treinamento...")
    results = model.train(
        data=dataset_yaml,
        epochs=20,
        imgsz=640,
        batch=8,
        device="cpu", # requisito do desafio
        patience=10,
        seed=42,
        workers=0, # evita problema de multiprocessing no Windows
        project=os.path.join(project_dir, "runs"),
        name="detect/train",
        verbose=True,
    )

    # Copiando o melhor modelo treinado para a raiz do projeto
    best_weights = os.path.join(results.save_dir, "weights", "best.pt")
    output_model = os.path.join(project_dir, "model.pt")

    if not os.path.exists(best_weights):
        raise FileNotFoundError(f"best.pt não encontrado após treinamento.")
    
    shutil.copy(best_weights, output_model)
    print(f"Modelo salvo em: {output_model}")


if __name__ == "__main__":
    main()
