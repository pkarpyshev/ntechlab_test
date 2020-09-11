def findMaxSubArray(a: list):
    max_sum = a[0]
    run_sum = a[0]
    start_ind = 0
    end_ind = 0
    for i in range(1, len(a)):
        if a[i] > run_sum + a[i]:
            run_sum = a[i]
            run_start = i
        else:
            run_sum += a[i]

        if run_sum > max_sum:
            start_ind = run_start
            end_ind = i
            max_sum = run_sum
    return a[start_ind:end_ind + 1]


if __name__ == '__main__':
    arr = [-2, -1, -1, -10, -5, -6, -7, -1]
    arr2 = [-2, 1, -3, 4, -1, 2, -1, -5, 4]
    print(findMaxSubArray(arr))
