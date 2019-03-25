/** 
 * Adapted from Alessandro Montanari's original code
 * 
*/

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/fcntl.h>
#include <time.h>
#include <signal.h>

int delay = 200;  // default delay in milliseconds between one sample and the other

int cnt = 0;
double value_v = 0;
double value_c = 0;
double start = 0;
struct timespec tv;
char *output_default_path = "/home/nvidia/values.csv";
char *output_filepath;

static volatile int keepRunning = 1;

void intHandler(int dummy) {
    keepRunning = 0;
}

int open_device(char *dev) {
    int fd = open(dev, O_RDONLY | O_NONBLOCK);

    if (fd < 0) {
        perror("open()");
        exit(1);
    }

    return fd;
}

double read_device(int dev) {
    char buf[31];

    lseek(dev, 0, 0);
    int n = read(dev, buf, 32);
    if (n > 0) {
        buf[n] = 0;
        char *o = NULL;
        return strtod(buf, &o);
    }
}

int main(int argc, char **argv) {
    signal(SIGINT, intHandler);     // Intercept ctrl+c and stop loop

    output_filepath = output_default_path;

    switch(argc)
    {
        // change default filepath
        case 3: 
            output_filepath = (char*) malloc(sizeof(char) * strlen(argv[2]));
            strcpy(output_filepath, argv[2])

        // change default delay
        case 2:
            delay = atoi(argv[1]);
    }

    int fd_voltage = open_device("/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/in_voltage0_input");
    int fd_current = open_device("/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/in_current0_input");

    FILE *f_output = fopen(output_filepath, "w");
    if (f_output == NULL) {
        perror("fopen()");
        exit(1);
    }

    printf("Writing on %s\n", output_filepath);
    printf("Ctrl+C to terminate sampling.\n");

    fprintf(f_output, "#time (seconds.nanoseconds), voltage (mV), current (mA)\n");

    while (keepRunning) {

        value_v = read_device(fd_voltage);
        value_c = read_device(fd_current);

        clock_gettime(CLOCK_REALTIME, &tv);
        fprintf(f_output, "%ld.%ld, %.2f, %.2f\n", tv.tv_sec, tv.tv_nsec, value_v, value_c);

        usleep(delay_ms * 1000);    // usleep uses delay in micro sencods
    }

    fclose(f_output);
    close(fd_voltage);
    close(fd_current);

    return 0;
}