#Naive approach
def max_prod_naive(arr):
    product = 0
    for i in range(len(arr)):
        for j in range(i+1,len(arr)):
            product = max(product,arr[i]*arr[j])
    return product