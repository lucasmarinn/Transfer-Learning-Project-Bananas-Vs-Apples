# Transfer-Learning-Project-Bananas-Vs-Apples

# 🍌 vs 🍎 - Classificação com Transfer Learning

Projeto de **Transfer Learning** usando **VGG16** para classificar imagens de bananas e maçãs.

## 📋 Sobre o Projeto
Implementação prática de Transfer Learning para classificação binária de imagens. Usa a arquitetura VGG16 pré-treinada no ImageNet e adapta para distinguir entre bananas e maçãs.

## 🎯 Objetivos
- Aplicar Transfer Learning em um problema real
- Demonstrar como modelos pré-treinados podem ser reutilizados
- Criar um classificador eficiente com poucos dados

## 🏗️ Arquitetura
- **Base:** VGG16 (pesos ImageNet)
- **Transfer Learning:** Congelamento total das camadas da VGG16
- **Novas camadas:**
  - Flatten
  - Dense(256, relu)
  - Dropout(0.5)
  - Dense(1, sigmoid) - saída binária

## 📁 Estrutura do Projeto

    projeto_transfer_learning/
    ├── bananas/                    # 100 fotos de bananas
    ├── macas/                      # 100 fotos de maçãs
    ├── train_model.py              # Treina o modelo
    ├── test_model.py               # Testa o modelo
    ├── requirements.txt            # Dependências
    ├── README.md                   # Esta documentação
    └── results/                    # Gerado automaticamente
        ├── modelo_final.h5         # Modelo treinado
        ├── historico_treino.npy    # Histórico do treino
        └── grafico_treinamento.png # Gráficos de performance


## 🚀 Como Executar

### Pré-requisitos
- Python 3.8+
- TensorFlow 2.x

### Instalação
```bash
# Clone o repositório
git clone https://github.com/seu-usuario/transfer-learning-project.git
cd transfer-learning-project

# Instale dependências
pip install -r requirements.txt
