THIRDPARTY_BUILD   := $(THIRDPARTY)/build
THIRDPARTY_INCLUDE := $(THIRDPARTY)/stow/include
THIRDPARTY_LIB     := $(THIRDPARTY)/stow/lib
THIRDPARTY_STOW    := $(THIRDPARTY)/stow

.PHONY: gtest

thirdparty-all: gtest

thirdparty-clean:
	rm -rf $(THIRDPARTY_STOW) $(THIRDPARTY_BUILD)

gtest: $(THIRDPARTY_LIB)/libgtest.la

$(THIRDPARTY_LIB)/libgtest.la:
	rm -rf $(THIRDPARTY_BUILD)
	mkdir $(THIRDPARTY_BUILD)
	@if [ ! -d $(THIRDPARTY_STOW) ]; then mkdir -p $(THIRDPARTY_STOW); fi
	cd $(THIRDPARTY_BUILD); $(THIRDPARTY)/gtest-1.6.0/configure --prefix=$(THIRDPARTY_STOW) CXX=$(CXX); make
#	cp $(THIRDPARTY_BUILD)/lib/.libs/{libgtest.a,libgtest.0.dylib} $(THIRDPARTY_STOW)/lib
#	cd $(THIRDPARTY_STOW)/lib; ln -s libgtest.0.dylib libgtest.dylib
	cp -r $(THIRDPARTY_BUILD)/lib $(THIRDPARTY_STOW)
	cp -r $(THIRDPARTY)/gtest-1.6.0/include $(THIRDPARTY_STOW)
