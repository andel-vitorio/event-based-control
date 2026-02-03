<div align="center">

# Controle Baseado em Eventos

**Andevaldo da Encarnação Vitório** _Mestre e Doutorando em Engenharia Elétrica_

**Orientador:** Prof. Dr. Iury Valente de Bessa  
_Universidade Federal do Amazonas (UFAM)_

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic-green?style=flat-square)](https://tede.ufam.edu.br/handle/tede/11191)
[![Status](https://img.shields.io/badge/Status-Active%20Dev-orange?style=flat-square)]()

</div>

---

> [!NOTE]
> **Status de Desenvolvimento** > O foco atual é a refatoração completa dos códigos da dissertação ([TEDE UFAM](https://tede.ufam.edu.br/handle/tede/11191)) e a implementação de novos cenários complexos para a tese, incluindo sistemas LPV, tolerância a falhas e segurança cibernética.

## 📋 Sobre o Projeto

Este repositório contém um framework robusto em Python para **síntese, simulação e análise de estratégias de Controle Acionado por Eventos (ETC)**. O projeto consolida a pesquisa iniciada no mestrado (2024-2025) e expandida no doutorado, focando na reprodutibilidade científica e na extensão para sistemas complexos.

---

## 📚 Produção Científica

### Dissertação de Mestrado (2025)

**Título:** _Controle baseado em eventos de sistemas lineares a parâmetros variantes sob distúrbios de energia limitada e atuadores saturantes_  
**Defesa:** 02 de Setembro de 2025  
**Link:** [Acessar no TEDE UFAM](https://tede.ufam.edu.br/handle/tede/11191)

> **Citação:** VITORIO, Andevaldo da Encarnação. **Controle baseado em eventos de sistemas lineares a parâmetros variantes sob distúrbios de energia limitada e atuadores saturantes**. 2025. 154 f. Dissertação (Mestrado em Engenharia Elétrica) – Universidade Federal do Amazonas, Manaus (AM), 2025.

<details>
<summary><strong>Ver Resumo / Abstract</strong> (Clique para expandir)</summary>

<br>

**Resumo:** Os Sistemas de Controle em Rede (NCS) têm papel essencial em aplicações industriais e tecnológicas... [Texto completo omitido para brevidade visual, mas incluído no contexto do documento original] ...A eficácia das abordagens é validada por meio de simulações numéricas.

**Abstract:** Networked Control Systems (NCS) play a crucial role in industrial and technological applications... [Full text omitted for visual brevity] ...The effectiveness of the proposed approaches is validated through numerical simulations.

</details>

---

## 🚀 Funcionalidades Principais

### Estratégias de Controle Baseado em Eventos

- **Síntese LMI Robusta:** Co-projeto de controladores e gatilhos via otimização convexa (CVXPY/MOSEK).
- **Mecanismos Avançados:**
  - **DETM:** Mecanismo de Acionamento Dinâmico (_Dynamic Event-Triggered Mechanism_).
  - **SETM / SETM\*:** Mecanismo de Acionamento Estáticos (_Static Event-Triggered Mechanism_).
  - **AETM:** Mecanismo de Acionamento Adaptativo (_Adaptive Event-Triggered Mechanism_).
  - **DAETM** Mecanismo de Acionamento Dinâmico-Adaptativo (_Dynamic-adaptive Event-Triggered Mechanism_).

### Cenários de Simulação

- **Sistemas LPV:** Modelagem de parâmetros variantes no tempo e incertezas politópicas.
- **Robustez e Segurança:**
  - Sistemas sob saturação de atuadores e perturbações externas.
  - **Tolerância a Falhas (FTC):** Compensação de falhas em tempo real.
  - **Cibersegurança:** Análise sob Ataques de Decepção (_Deception Attacks_).
- **Aplicações:** Controle de temperatura (HVAC) e Conversores DC-DC.

---

## 📂 Estrutura do Repositório

O projeto opera como um pacote Python modular (`event_based_control`).

### `optimization/`

Núcleo de síntese dos controladores.

- `DisturbedSaturatedPETC.py`: Implementação das classes `DETM` e `SETM` considerando perturbação e saturação, além das rotinas de otimização LMI.

### `PETC for LIT Systems/`

Notebooks para sistemas Lineares Invariantes no Tempo (LIT).

- `2 - Fault Tolerance.ipynb`: Estudos sobre tolerância a falhas.
- `3 - HVAC Under Disturbances.ipynb`: Aplicação em sistemas térmicos prediais.
- `4/5 - Systems under Saturation...`: Análise de ataques e saturação.

### `PETC for LPV Systems/`

Foco em sistemas Lineares com Parâmetros Variantes.

- `petc_simulation.py`: Rotinas de malha fechada para LPV.
- `Results/`: Logs de experimentos comparativos (Síncrono, SETM\*, DAETM).

### `Utils/`

Bibliotecas auxiliares (_Backend_).

- `DynamicSystem.py`: Engines de simulação (`SimulationEngine`), amostradores e plantas.
- `Numeric.py`: Métodos numéricos (Runge-Kutta 5ª ordem) e geometria de conjuntos.
- `Graphs.py` & `Tex.py`: Ferramentas de visualização e exportação para LaTeX.

---

## 🛠 Instalação e Configuração

O projeto utiliza um `Makefile` para orquestrar o ambiente.

### Pré-requisitos

- **Solvers:** Recomenda-se o **MOSEK** (licença acadêmica) para estabilidade numérica nas LMIs.
- **Python:** 3.10 ou superior.

### Comandos de Instalação

No terminal, na raiz do projeto:

**1. Modo Desenvolvimento (Recomendado)**
Instala as dependências e linka os módulos locais (`Utils`, `optimization`) para edição em tempo real.

```bash
make dev

```

**2. Instalação Padrão**
Apenas para execução dos notebooks existentes.

```bash
make install

```
