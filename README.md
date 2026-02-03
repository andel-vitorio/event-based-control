# Controle Baseado em Eventos

> **Autor:** Andevaldo da Encarnação Vitório  
> **Orientador:** Prof. Dr. Iury Valente de Bessa  
> **Instituição:** Universidade Federal do Amazonas (UFAM)  
> **Curso:** Mestrado/Doutorado em Engenharia Elétrica

---

Este projeto foi realizado durante o meu mestrado (2024-2025) e está sendo continuado no doutorado. Atualmente, estou trabalhando na refatoração de todos os códigos utilizados para obter os resultados da minha dissertação de mestrado, disponível em: [TEDE UFAM](https://tede.ufam.edu.br/handle/tede/11191).

Este repositório contém uma estrutura em Python para o projeto (síntese) e simulação de estratégias de controle baseadas em eventos. O foco atual inclui o Controle Acionado po Eventos (ETC) aplicado a sistemas Lineares com Parâmetros Variantes (LPV), mas a arquitetura permite a extensão para outros tipos de sistemas e mecanismos de acionamento, considerando perturbações externas e saturação nos atuadores.

## Funcionalidades Principais

- **Síntese LMI**: Algoritmos baseados em otimização convexa (Desigualdades Matriciais Lineares) para projetar co-projetar o controlador e o mecanismo de disparo.
  - **DETM**: Mecanismo de Disparo de Eventos Dinâmico (_Dynamic Event-Triggered Mechanism_).
  - **SETM**: Mecanismo de Disparo de Eventos Estático (_Static Event-Triggered Mechanism_).
- **Simulação de Sistemas Dinâmicos**: Um motor de simulação (`SimulationEngine`) capaz de lidar com sistemas lineares e LPVs.
- **Sistemas LPV**: Suporte para definição de sistemas com parâmetros variantes no tempo, incluindo limites de variação e incertezas politópicas.
- **Ferramentas de Análise**: Utilitários para plotagem de gráficos, formatação LaTeX e cálculos geométricos (elipsoides, poliedros).

## Estrutura do Projeto

### `optimization/`

Contém os módulos de otimização para o projeto dos controladores.

- `DisturbedSaturatedPETC.py`: Implementa as funções `detm_synthesis` e `setm_synthesis` usando CVXPY para resolver os problemas de otimização. Também define as classes `DETM` e `SETM` usadas na simulação.

### `PETC for LPV Systems/`

Scripts e notebooks focados na aplicação e simulação.

- `petc_simulation.py`: Contém as funções de malha fechada (`closed_loop_detm`, `closed_loop_setm`) que integram a planta, o controlador e o mecanismo de eventos.

### `PETC for LIT Systems/`

Scripts e notebooks focados em sistemas Lineares Invariantes no Tempo (LIT).

### `PETC for DC Converters/`

Aplicações específicas para conversores DC-DC.

### `Utils/`

Bibliotecas auxiliares para o funcionamento do framework.

- `DynamicSystem.py`: Classes principais como `StateSpace` (para modelagem da planta), `SimulationEngine` (motor de simulação), `Sampler` e `GainScheduledController`.
- `Numeric.py`: Funções matemáticas, integração numérica (Runge-Kutta de 5ª ordem), operações com conjuntos e geometria.
- `Graphs.py`: Funções para geração de gráficos (evolução de estados, intervalos entre eventos, planos de fase).
- `Tex.py`: Utilitários para exportação e formatação de resultados em LaTeX.

## Instalação e Requisitos

Este projeto foi estruturado como um pacote Python para facilitar a importação dos módulos compartilhados nos notebooks de estudo. Para rodar os exemplos sem problemas, é necessário realizar a instalação das dependências.

### Dependências Principais

- **CVXPY**: Utilizado como parser para a modelagem dos problemas de otimização convexa (LMIs).
- **MOSEK**: Solver de otimização de alta performance, altamente recomendado para os problemas tratados neste projeto.
  - _Nota_: O MOSEK requer uma licença (acadêmica ou comercial) para funcionar. Certifique-se de que a licença esteja configurada corretamente em seu ambiente.
- **Bibliotecas Científicas**: `numpy`, `scipy` para cálculos numéricos e `matplotlib` para visualização.

### Configuração do Ambiente

O projeto inclui um `Makefile` para automatizar a instalação. No terminal, navegue até a raiz do repositório e execute um dos comandos abaixo:

1.  **Instalação em Modo de Desenvolvimento (Recomendado)**:
    Utilize este comando se você pretende modificar os códigos fonte em `Utils` ou `optimization` e quer que as alterações sejam refletidas imediatamente nos notebooks.

    ```bash
    make dev
    ```

2.  **Instalação Padrão**:
    Para apenas utilizar os módulos como estão.
    ```bash
    make install
    ```
