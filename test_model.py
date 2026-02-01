"""
TESTE do modelo treinado - Bananas vs Maçãs
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ============ CONFIGURAÇÕES ============
MODEL_PATH = 'results/modelo_final.h5'
CLASS_NAMES = {0: '🍌 BANANA', 1: '🍎 MAÇÃ'}  # Baseado no mapeamento do treino

# ============ VERIFICAR MODELO ============
print("🔍 Verificando se o modelo existe...")

if not os.path.exists(MODEL_PATH):
    print(f"❌ Modelo não encontrado em: {MODEL_PATH}")
    print("💡 Execute primeiro: python train_model.py")
    exit()

print("✅ Modelo encontrado!")
print("📥 Carregando modelo...")

# Carregar modelo
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Modelo carregado!")

# ============ FUNÇÃO PARA TESTAR IMAGEM ============
def testar_imagem(caminho_imagem):
    """
    Testa uma imagem e mostra o resultado
    """
    if not os.path.exists(caminho_imagem):
        print(f"❌ Arquivo não encontrado: {caminho_imagem}")
        return None
    
    try:
        # Carregar e preparar imagem
        img = image.load_img(caminho_imagem, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Fazer predição
        predicao = model.predict(img_array, verbose=0)[0][0]
        
        # Interpretar resultado
        classe_idx = 1 if predicao > 0.5 else 0
        classe_nome = CLASS_NAMES[classe_idx]
        confianca = predicao if classe_idx == 1 else 1 - predicao
        
        # Mostrar resultado
        print("\n" + "="*40)
        print(f"📸 Imagem: {os.path.basename(caminho_imagem)}")
        print(f"🎯 Resultado: {classe_nome}")
        print(f"📊 Confiança: {confianca:.2%}")
        print(f"🔢 Valor bruto: {predicao:.4f}")
        print("="*40)
        
        # Plotar imagem com resultado
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f"{classe_nome}\nConfiança: {confianca:.2%}", 
                  fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        
        # Adicionar cor baseada no resultado
        cor_fundo = 'lightgreen' if confianca > 0.7 else 'lightcoral'
        plt.gcf().patch.set_facecolor(cor_fundo)
        
        plt.tight_layout()
        plt.show()
        
        return classe_nome, confianca
        
    except Exception as e:
        print(f"❌ Erro ao processar imagem: {e}")
        return None

# ============ FUNÇÃO PARA TESTAR VÁRIAS IMAGENS ============
def testar_pasta(pasta, classe_esperada):
    """
    Testa várias imagens de uma pasta
    """
    if not os.path.exists(pasta):
        print(f"❌ Pasta não encontrada: {pasta}")
        return
    
    print(f"\n🧪 Testando pasta: {pasta}")
    print(f"📂 Esperado: {classe_esperada}")
    
    imagens = [f for f in os.listdir(pasta) 
               if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if not imagens:
        print("❌ Nenhuma imagem encontrada na pasta")
        return
    
    # Testar até 5 imagens
    imagens = imagens[:5]
    acertos = 0
    
    for img_nome in imagens:
        caminho = os.path.join(pasta, img_nome)
        resultado = testar_imagem(caminho)
        
        if resultado:
            classe_predita, confianca = resultado
            if classe_esperada.lower() in classe_predita.lower():
                acertos += 1
    
    # Estatísticas
    print(f"\n📊 Estatísticas da pasta '{os.path.basename(pasta)}':")
    print(f"✅ Testadas: {len(imagens)} imagens")
    print(f"🎯 Acertos: {acertos}/{len(imagens)}")
    print(f"📈 Taxa de acerto: {(acertos/len(imagens))*100:.1f}%")

# ============ MENU INTERATIVO ============
def menu_principal():
    """
    Menu principal interativo
    """
    while True:
        print("\n" + "="*50)
        print("🧪 TESTADOR DE MODELO - BANANAS vs MAÇÃS")
        print("="*50)
        print("\nEscolha uma opção:")
        print("1. Testar uma imagem específica")
        print("2. Testar várias bananas")
        print("3. Testar várias maçãs")
        print("4. Ver estatísticas gerais")
        print("5. Sair")
        
        opcao = input("\n👉 Digite sua opção (1-5): ").strip()
        
        if opcao == "1":
            # Testar imagem específica
            caminho = input("Digite o caminho da imagem: ").strip()
            testar_imagem(caminho)
            
        elif opcao == "2":
            # Testar pasta de bananas
            if os.path.exists('bananas'):
                testar_pasta('bananas', 'banana')
            else:
                print("❌ Pasta 'bananas' não encontrada")
                
        elif opcao == "3":
            # Testar pasta de maçãs
            if os.path.exists('macas'):
                testar_pasta('macas', 'maçã')
            else:
                print("❌ Pasta 'macas' não encontrada")
                
        elif opcao == "4":
            # Estatísticas gerais
            if os.path.exists('results/historico_treino.npy'):
                historico = np.load('results/historico_treino.npy', allow_pickle=True).item()
                print("\n📊 ESTATÍSTICAS DO TREINAMENTO:")
                print(f"Acurácia final treino: {historico['accuracy'][-1]:.2%}")
                print(f"Acurácia final validação: {historico['val_accuracy'][-1]:.2%}")
                print(f"Épocas treinadas: {len(historico['accuracy'])}")
            else:
                print("ℹ️  Execute o treinamento primeiro para ver estatísticas")
                
        elif opcao == "5":
            print("\n👋 Até logo!")
            break
            
        else:
            print("❌ Opção inválida! Digite 1, 2, 3, 4 ou 5.")

# ============ EXECUTAR ============
if __name__ == "__main__":
    print("🚀 Iniciando testador de modelo...")
    menu_principal()