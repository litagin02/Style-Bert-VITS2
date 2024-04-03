.PHONY: build
build:
	docker-compose -f compose.api.yml build --build-arg UID="`id -u`" --build-arg GID="`id -g`"

.PHONY: run
run:
	docker-compose -f compose.api.yml up
