#include <stdio.h>
#include <string.h>

struct mystruct
{
    char buffer_two[52], buffer_one[48];
    int mazil;
};

void test_case(char *input)
{
    struct mystruct mystruct1;
    float score = 0.0;
    mystruct1.mazil = 11;
    strcpy(mystruct1.buffer_one, input);

    if (mystruct1.mazil == 11)
        score += 0.5;

    if (mystruct1.mazil > 8316 && mystruct1.mazil < 8337)
        score += 0.5;

    printf("%.2f", score);
}

int main()
{
    char input[256];
    if (fgets(input, sizeof(input), stdin))
        test_case(input);
    else
        printf("\n[FAILURE] No input provided.\n");

    return 0;
}