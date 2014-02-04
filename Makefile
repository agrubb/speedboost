.PHONY: all test clean distclean
.DEFAULT_GOAL  = all

ROOT       := $(shell pwd)
THIRDPARTY := $(ROOT)/thirdparty

include thirdparty/Makefile

CXX      = g++
OPTS     = -O3 -g -fPIC -fopenmp
CXXFLAGS = -Wall $(OPTS) -I$(THIRDPARTY_INCLUDE) -I/usr/local/include/ -I/usr/include/ImageMagick/ `pkg-config --cflags protobuf`
LDFLAGS  = -fopenmp -Llib/ -L$(THIRDPARTY_LIB)/
LDLIBS   = -lspeedboost -lgflags -lMagick++ `pkg-config --libs protobuf`

LIBRARY = lib/libspeedboost.a

ALL_MODULES = \
src \
test

DISABLED_MODULES = 

ENABLED_MODULES  = 

MODULES        = $(filter-out $(DISABLED_MODULES),$(ALL_MODULES))
ifneq ($(ENABLED_MODULES),)
  MODULES := $(filter $(ENABLED_MODULES),$(MODULES))
endif


$(foreach module,$(MODULES),$(eval include $(module)/Makefile))

PROTO_CC  = $(PROTO_SRC:.proto=.pb.cc)
PROTO_H   = $(PROTO_SRC:.proto=.pb.h)
ALL_PROTO = $(PROTO_CC) $(PROTO_H)
SRC += $(PROTO_CC)

ALL_SRC  = $(SRC) $(TEST_SRC) $(MAIN_SRC)
ALL_OBJ  = $(addprefix obj/,$(ALL_SRC:.cc=.o))
ALL_DEP  = $(addprefix obj/,$(ALL_SRC:.cc=.d))

OBJ      = $(addprefix obj/,$(SRC:.cc=.o))
TEST_OBJ = $(addprefix obj/,$(TEST_SRC:.cc=.o))
MAIN_OBJ = $(addprefix obj/,$(MAIN_SRC:.cc=.o))

PROGRAM_NAME = bin/$(basename $(notdir $(1)))
$(foreach main,$(MAIN_OBJ),$(eval $(call PROGRAM_NAME,$(main)): $(main) $(LIBRARY)))
PROGRAMS += $(foreach main,$(MAIN_OBJ),$(call PROGRAM_NAME,$(main)))


all: thirdparty-all protos $(PROGRAMS)

clean:
	rm -rf obj lib $(PROGRAMS)

distclean: clean thirdparty-clean

bin/check: LDLIBS += $(THIRDPARTY_LIB)/.libs/libgtest.a
bin/check: $(TEST_OBJ) $(LIBRARY)

test: thirdparty-all bin/check
	@if [ ! -d test-output ]; then mkdir -p test-output; fi
	@./bin/check --test_output_directory=test-output
	@rm -rf test-output

protos: $(ALL_PROTO)

$(LIBRARY): $(OBJ)
	@if [ ! -d $(dir $@) ]; then mkdir -p $(dir $@); fi
	$(AR) rcs $@ $(OBJ)

$(PROGRAMS): %:
	@if [ ! -d $(dir $@) ]; then mkdir -p $(dir $@); fi
	$(CXX) $(LDFLAGS) -o $@ $(filter %.o,$^) $(LDLIBS)

$(PROTO_CC): %.pb.cc: %.proto
	cd $(dir $@); protoc --cpp_out=. $(notdir $<)

$(TEST_OBJ): CXXFLAGS += -Isrc/

$(ALL_OBJ): obj/%.o: %.cc
	@if [ ! -d $(dir $@) ]; then mkdir -p $(dir $@); fi
	$(CXX) -MMD $(CXXFLAGS) -o $@ -c $<

-include $(ALL_DEP)
