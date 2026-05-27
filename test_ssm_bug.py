import torch
from mamba_ssm import Mamba3

def run_smem_bug_test():
    # 1. Configurar el entorno en GPU (CUDA)
    device = "cuda"
    dtype = torch.bfloat16  # Precisión media (bfloat16) como en el issue
    
    # 2. Parámetros del bloque Mamba-3
    batch, seqlen, dim = 2, 512, 768
    
    # IMPORTANTE: Aquí configuramos los valores que sobre-asignan la memoria compartida
    model = Mamba3(
        d_model=dim,
        d_state=128,              # Tamaño del estado del SSM
        headdim=64,               # Dimensión de las cabezas del SSM
        is_mimo=True,             # Forzamos modo MIMO (Multi-Input Multi-Output)
        mimo_rank=16,              # Rango de la proyección MIMO
        chunk_size=32,            # ¡ESTO PROVOCA EL ERROR! (Excede el límite de 64/mimo_rank)
        dtype=dtype
    ).to(device)

    # 3. Crear datos ficticios que requieran gradiente para forzar el backward pass
    #print("Creando tensores de entrada...")
    x = torch.randn(batch, seqlen, dim, device=device, dtype=dtype, requires_grad=True)

    # 4. Forward Pass (Paso hacia adelante)
    #print("Ejecutando Forward Pass...")
    out = model(x)
    
    # 5. Backward Pass (Paso hacia atrás)
    # Aquí es donde el compilador TileLang calcula la derivada y solicita 140.4 KB de Smem
    #print("Ejecutando Backward Pass (aquí debería fallar si tu GPU tiene < 140 KB de Smem)...")
    loss = out.float().sum()
    loss.backward()
    
    print("Se logró minimizar el tamaño de memoria compartida")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: Necesitas una GPU compatible con CUDA activa para correr este test.")
    else:
        run_smem_bug_test()
