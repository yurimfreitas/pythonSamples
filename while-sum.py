num_int = int(input("Enter a positive number: "))
i = num_int
sum_all = 0

while i > 0:
    sum_all = sum_all + i
    i -= 1

print("The number choosed was: "+str(num_int))
print("The sum is: "+str(sum_all))
