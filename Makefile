# Load environment variables from .env
-include .env
export

.PHONY: build deploy stop logs clean fix-daemon attach help

## build: Build the vision pipeline image
build:
	docker compose -f docker-compose.yml build

## deploy: Start the vision pipeline container in background
## (xhost est nécessaire ici car speed_det ouvre une fenêtre de debug dès le lancement,
##  contrairement au wrapper caméra où l'affichage n'est utilisé que via 'make rviz')
deploy:
	xhost +local:docker
	docker compose -f docker-compose.yml up -d --force-recreate
	@echo "Vision pipeline deployment started. Use 'make logs' to view."

## logs: Follow live logs from the deployment container
logs:
	docker compose -f docker-compose.yml logs -f

## stop: Stop the deployment container
stop:
	docker compose -f docker-compose.yml stop

## clean: Remove all local ROS 2 build artifacts and docker containers
clean:
	docker compose -f docker-compose.yml down --remove-orphans
	rm -rf build/ install/ log/ Log/
	@echo "Cleanup complete."

## fix-daemon: Kill a possibly stale local ros2-daemon (mismatched RMW_IMPLEMENTATION)
## A lancer si 'ros2 topic list' ne montre pas les topics attendus alors que
## les conteneurs tournent correctement (cf. debug session du 2026-07-28).
fix-daemon:
	@pkill -9 -f "ros2-daemon" 2>/dev/null && echo "ros2-daemon (host) arrêté, redémarrera automatiquement." || echo "Aucun ros2-daemon local trouvé."

## attach: Open an interactive bash shell inside the running deployment container
attach:
	@echo "Attaching interactive shell to 'ros2_vision_container'..."
	docker exec -it ros2_vision_container bash

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^##' Makefile | sed -e 's/## //g' -e 's/: /:	/g'