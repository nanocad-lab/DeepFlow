#CC     = /usr/bin/gcc-3.4

#CFLAGS = -I. $(DEFS) -O3 -Wno-deprecated
CFLAGS = -I. $(DEFS) -g -DTECHNEW

##AR     = ar
##RANLIB = ranlib

ROUTER_MODEL = SIM_router.c SIM_arbiter.c SIM_crossbar.c router.c SWVC.c XBAR.c INBUF.c OUTBUF.c CLKCTRL.c NONPR.c
POWER_MODEL	 = SIM_router_power.c router_power.c
LINK_MODEL   = SIM_link.c
CLOCK_MODEL  = SIM_clock.c
AREA_MODEL   = SIM_router_area.c router_area.c
ARRAY_MODEL  = SIM_array_l.c SIM_array_m.c SIM_cam.c
ALU_MODEL    = SIM_ALU.c
OTHER_MODEL  = SIM_misc.c SIM_permu.c
STATIC_MODEL = SIM_static.c
MISC_SRC     = SIM_util.c SIM_time.c

SRCS         = $(ARRAY_MODEL) $(ROUTER_MODEL) $(POWER_MODEL) $(ALU_MODEL) $(OTHER_MODEL) \
               $(STATIC_MODEL) $(LINK_MODEL) $(CLOCK_MODEL) $(MISC_SRC) $(AREA_MODEL) 

OBJS      = $(SRCS:.c=.o)


all: 
	make orion_router

TEST_SRCS   = orion_router.c  test_permu.c test_alu.c dump_para.c
TEST_EXEC   = $(TEST_SRCS:.c=)
MODULE_SRCS = SIM_router_power.c router_power.c router_area.c
ALL_SRCS    = $(SRCS) $(TEST_SRCS) $(MODULE_SRCS) 
ALL_OBJS    = $(ALL_SRCS:.c=.o)

orion_router: orion_router.o router_power.o router_area.o SIM_router_power.o $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm 

test_permu: test_permu.o $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm 

test_alu: test_alu.o $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

dump_para: dump_para.o $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

clean:
	$(RM) $(ALL_OBJS) $(TEST_EXEC) *~ *.bak core

