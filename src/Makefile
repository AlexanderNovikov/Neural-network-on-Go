NAME		= nn-go

GOVERSION	= go1.13.1
GOFLAGS		= GOOS=$(GOOS) GOARCH=$(GOARCH)
GOROOT      = $(HOME)/$(GOVERSION)
GO			= GODEBUG=sbrk=1 $(GOROOT)/bin/go
VERSION		= $(shell git log --pretty=format:%h --abbrev=10 | head -1)
GOOPTS		= -ldflags "$(GOLDFLAGS)"

PKG			= $(NAME)_$(VERSION)
ARCHDIR		= $(GOPATH)/../pkg/$(GOOS)_$(GOARCH)
PKGDIR		= $(ARCHDIR)/$(PKG)

# path to this file
ROOT        = $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
GOPATH      = $(realpath $(ROOT)/)
export GOROOT
export GOPATH

# OSX requires GNU tar
TAR	= $(shell which tar)
ifeq ($(shell uname),Darwin)
	TAR = $(shell which tar)
endif

.PHONY: buildpkg
buildpkg:
	@mkdir -p $(PKGDIR)/bin
	@$(GOFLAGS) $(GO) build $(GOOPTS) -o $(PKGDIR)/bin/$(NAME)
	@cd $(ARCHDIR) && $(TAR) cp -f $(PKG).tar.gz -z $(PKG) && rm -rf $(PKG)
	@echo "Created $(ARCHDIR)/$(PKG).tar.gz"

.PHONY: distpkg
distpkg:
	@$(MAKE) GOOS=linux GOARCH=amd64 buildpkg
