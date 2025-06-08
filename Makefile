.PHONY: help build up down logs test lint format clean backup

# Default target
help:
	@echo "🚀 Crypto Trading AI System"
	@echo ""
	@echo "Available commands:"
	@echo "  build     - Build all Docker images"
	@echo "  up        - Start all services"
	@echo "  down      - Stop all services"
	@echo "  logs      - Show logs for all services"
	@echo "  test      - Run tests"
	@echo "  lint      - Run linting (ruff)"
	@echo "  format    - Format code (black + ruff)"
	@echo "  clean     - Clean up Docker resources"
	@echo "  backup    - Backup database"
	@echo ""

# Docker operations
build:
	@echo "🏗️ Building Docker images..."
	docker compose build

up:
	@echo "🚀 Starting services..."
	docker compose up -d
	@echo "✅ Services started!"
	@echo "📊 Dashboard: http://localhost:8501"
	@echo "🔄 Airflow: http://localhost:8080"

down:
	@echo "🛑 Stopping services..."
	docker compose down

logs:
	@echo "📋 Showing logs..."
	docker compose logs -f

# Testing
test:
	@echo "🧪 Running tests..."
	docker compose exec core-app python -m pytest tests/ -v

test-local:
	@echo "🧪 Running tests locally..."
	cd core-app && python -m pytest ../tests/ -v

# Code quality
lint:
	@echo "🔍 Running linting..."
	docker compose exec core-app ruff check src/
	docker compose exec core-app ruff check tests/

format:
	@echo "✨ Formatting code..."
	docker compose exec core-app black src/ tests/
	docker compose exec core-app ruff --fix src/ tests/

format-local:
	@echo "✨ Formatting code locally..."
	black core-app/src/ tests/
	ruff --fix core-app/src/ tests/

# Maintenance
clean:
	@echo "🧹 Cleaning up..."
	docker compose down -v
	docker system prune -f
	docker volume prune -f

backup:
	@echo "💾 Backing up database..."
	docker compose exec postgres pg_dump -U trade tradedb > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "✅ Database backed up!"

# Development helpers
shell-core:
	@echo "🐚 Opening shell in core-app..."
	docker compose exec core-app bash

shell-db:
	@echo "🐚 Opening PostgreSQL shell..."
	docker compose exec postgres psql -U trade -d tradedb

dashboard-logs:
	@echo "📊 Showing dashboard logs..."
	docker compose logs -f core-app

airflow-logs:
	@echo "🔄 Showing Airflow logs..."
	docker compose logs -f airflow-webserver airflow-scheduler

# Quick setup
setup: build up
	@echo "⏳ Waiting for services to start..."
	sleep 30
	@echo "✅ Setup complete!"
	@echo "📊 Dashboard: http://localhost:8501"
	@echo "🔄 Airflow: http://localhost:8080 (airflow/airflow)"

# Reset everything
reset: down clean
	@echo "🔄 Resetting system..."
	docker compose up -d
	@echo "✅ System reset complete!"