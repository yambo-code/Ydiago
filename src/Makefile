include make.inc

TARGET 		 :=  libdiago.a

#### make start here

SUBDIR = matrix elpa SL 


INCS = $(wildcard *.h $(foreach fd, $(SUBDIR), $(fd)/*.h))
SRCS = $(wildcard *.c $(foreach fd, $(SUBDIR), $(fd)/*.c))
OBJS = $(addprefix ./, $(SRCS:c=o)) 

INC_DIRS += $(ELPA_INC)

PHONY := $(TARGET)
$(TARGET): $(OBJS) 
	$(AR) rcs ./$@ $(OBJS) 

%.o: %.c $(INCS) make.inc
	$(CC) $(CFLAGS) -c $< $(INC_DIRS) $(OPENMP_FLAGS) -o $@


PHONY += clean
clean:
	rm ./$(TARGET) $(OBJS)

PHONY += echoes
echoes:
	@echo "INC files: $(INCS)"
	@echo "SRC files: $(SRCS)"
	@echo "OBJ files: $(OBJS)"
	@echo "INC DIR: $(INC_DIRS)"


.PHONY = $(PHONY)
