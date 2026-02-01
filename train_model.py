import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# ============ 1. VERIFICAR PASTAS ============
print("🔍 Verificando pastas...")

PASTA_BANANAS = 'bananas'
PASTA_MACAS = 'macas'

if not os.path.exists(PASTA_BANANAS):
    print(f"❌ Pasta '{PASTA_BANANAS}' não encontrada")
    exit()

if not os.path.exists(PASTA_MACAS):
    print(f"❌ Pasta '{PASTA_MACAS}' não encontrada")
    exit()

# Contar fotos
num_bananas = len([f for f in os.listdir(PASTA_BANANAS) if f.endswith(('.jpg', '.png', '.jpeg'))])
num_macas = len([f for f in os.listdir(PASTA_MACAS) if f.endswith(('.jpg', '.png', '.jpeg'))])

print(f"🍌 Fotos de bananas: {num_bananas}")
print(f"🍎 Fotos de maçãs: {num_macas}")

# ============ 2. CRIAR ESTRUTURA TEMPORÁRIA ============
print("\n📁 Criando estrutura para o Keras...")

import shutil
import tempfile

# Criar diretório temporário
temp_dir = tempfile.mkdtemp()
print(f"📂 Diretório temporário: {temp_dir}")

# Criar pasta 'data' com subpastas 'bananas' e 'macas'
data_dir = os.path.join(temp_dir, 'data')
os.makedirs(os.path.join(data_dir, 'bananas'), exist_ok=True)
os.makedirs(os.path.join(data_dir, 'macas'), exist_ok=True)

# Copiar TODAS as fotos mantendo estrutura
print("📤 Copiando fotos...")

for img in os.listdir(PASTA_BANANAS):
    if img.lower().endswith(('.jpg', '.png', '.jpeg')):
        shutil.copy2(
            os.path.join(PASTA_BANANAS, img),
            os.path.join(data_dir, 'bananas', img)
        )

for img in os.listdir(PASTA_MACAS):
    if img.lower().endswith(('.jpg', '.png', '.jpeg')):
        shutil.copy2(
            os.path.join(PASTA_MACAS, img),
            os.path.join(data_dir, 'macas', img)
        )

print("✅ Fotos copiadas!")

# ============ 3. CONFIGURAR DATA GENERATOR ============
IMG_SIZE = (150, 150)
BATCH_SIZE = 16

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 20% validação
)

print("\n📊 Carregando imagens...")

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

print(f"✅ Classes: {train_generator.class_indices}")
print(f"📈 Treino: {train_generator.samples} imagens")
print(f"📉 Validação: {val_generator.samples} imagens")


# ============ 4. TRANSFER LEARNING COM VGG16 ============
print("\n🏗️ Configurando Transfer Learning com VGG16...")

# Carregar VGG16 pré-treinada
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)
)

# Congelar todas as camadas da VGG16
for layer in base_model.layers:
    layer.trainable = False

print("✅ VGG16 carregada e congelada")

# Adicionar nossas camadas
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)  # Binary classification

# Criar modelo final
model = Model(inputs=base_model.input, outputs=output)

# Compilar
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("📋 Resumo do modelo:")
model.summary()

# ============ 5. TREINAR ============
print("\n🚀 Iniciando treinamento...")

EPOCHS = 10

history = model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
    validation_data=val_generator,
    validation_steps=max(1, val_generator.samples // BATCH_SIZE),
    epochs=EPOCHS,
    verbose=1
)

# ============ 6. SALVAR RESULTADOS ============
print("\n💾 Salvando resultados...")

# Criar pasta 'results' se não existir
os.makedirs('results', exist_ok=True)

# Salvar modelo
model.save('results/modelo_final.h5')
print("✅ Modelo salvo: results/modelo_final.h5")

# Salvar histórico de treino
np.save('results/historico_treino.npy', history.history)
print("✅ Histórico salvo")

# ============ 7. GRÁFICOS ============
print("\n📈 Gerando gráficos...")

plt.figure(figsize=(12, 4))

# Gráfico 1: Acurácia
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia durante treinamento')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True)

# Gráfico 2: Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Loss durante treinamento')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('results/grafico_treinamento.png', dpi=100)
plt.show()

# ============ 8. RESULTADOS FINAIS ============
print("\n" + "="*50)
print("🎯 RESULTADOS FINAIS")
print("="*50)

final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print(f"🍌🍎 Dataset: {train_generator.samples + val_generator.samples} imagens")
print(f"📈 Acurácia no TREINO: {final_train_acc:.2%}")
print(f"📈 Acurácia na VALIDAÇÃO: {final_val_acc:.2%}")
print(f"📉 Loss no TREINO: {final_train_loss:.4f}")
print(f"📉 Loss na VALIDAÇÃO: {final_val_loss:.4f}")
print(f"🔢 Mapeamento classes: {train_generator.class_indices}")

print("\n💡 Próximo passo: Execute 'test_model.py' para testar o modelo!")
print("="*50)

# Limpar diretório temporário
shutil.rmtree(temp_dir)
print(f"\n🧹 Diretório temporário removido: {temp_dir}")