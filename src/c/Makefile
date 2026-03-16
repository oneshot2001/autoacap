PROG1 = autoacap
OBJS1 = app.o
PKGS = vdostream gio-2.0 glib-2.0

CFLAGS += -O3 -Wall -Werror -Wno-unused-result
CFLAGS += $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config --cflags $(PKGS))
LDLIBS += $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config --libs $(PKGS))
LDLIBS += -llarod -lm

all: $(PROG1)

$(PROG1): $(OBJS1)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

clean:
	rm -f $(PROG1) $(OBJS1)
