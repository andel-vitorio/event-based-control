.PHONY: help save sync tag

# Configurações padrão
MSG ?= "Atualização automática"
BRANCH ?= main

help:
	@echo "Comandos disponíveis:"
	@echo "  make save MSG='Sua mensagem'   - Adiciona e commita alterações localmente"
	@echo "  make sync MSG='Sua mensagem'   - Commita e envia para o GitHub (push)"
	@echo "  make tag v=1.0.0               - Cria uma tag de versão e envia para o GitHub"

save:
	git add .
	git commit -m "$(MSG)" || echo "Nenhuma alteração para salvar."

sync: save
	git push origin $(BRANCH)

tag:
	@if [ -z "$(v)" ]; then echo "Erro: especifique a versão. Ex: make tag v=1.0.0"; exit 1; fi
	git tag -a v$(v) -m "Versão $(v)"
	git push origin v$(v)