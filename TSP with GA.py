import random #generating random numbers
import math #mathematical operations
import matplotlib.pyplot as plt #visualizing the results.


#get cities information.
def getUserCities():
    cities = []
    num_cities = int(input("Please enter the number of cities: "))

    for i in range(num_cities):
        city_name = input("Please enter the name of city {}: ".format(i+1))
        city_x = float(input("Please enter the longitude of city {}: ".format(i+1)))
        city_y = float(input("Please enter the latitude of city {}: ".format(i+1)))

        cities.append([city_name, city_x, city_y])

    return cities


# calculating distance of the cities
def calcDistance(cities):
    total_sum = 0 #distance between cities.
    for i in range(len(cities) - 1): #loop over the indices of the cities list.
        cityA = cities[i] #the current city
        cityB = cities[i + 1] #the next city

        #calculates the distance between cityA and cityB.
        d = math.sqrt(
            math.pow(cityB[1] - cityA[1], 2) + math.pow(cityB[2] - cityA[2], 2)
        )

        total_sum += d

    cityA = cities[0] #the first city
    cityB = cities[-1] #the last city
    #calculates the distance between the first and last city
    d = math.sqrt(math.pow(cityB[1] - cityA[1], 2) + math.pow(cityB[2] - cityA[2], 2))

    total_sum += d

    return total_sum  # the total distance of the TSP tour.


# selecting the population
def selectPopulation(cities, size): 
    population = [] #the population of TSP tours.

    for i in range(size): #the initial population will consist of 500 randomly generated TSP tours.
        c = cities.copy() #a copy of the cities 
        random.shuffle(c) #creating a new permutation.
        distance = calcDistance(c) #total distance of the TSP tour.
        population.append([distance, c]) 
    fitest = sorted(population)[0] #Tour with the shortest distance

    return population, fitest


# the genetic algorithm
def geneticAlgorithm(
    population, #The initial population
    lenCities, #The number of cities
    tournament_size, #The size of the tournament selection
    MUTATION_RATE, #The probability of mutation
    CROSSOVER_RATE, #The probability of crossover
):
    gen_number = 0 #keep track of the current generation number.
    for i in range(200): #loop that iterates 200 times.
        new_population = [] 

        # selecting two of the best options we have (elitism)
        new_population.append(sorted(population)[0]) #the fittest tour 
        new_population.append(sorted(population)[1]) #the second fittest tour 

        for i in range(int((len(population) - 2) / 2)):
            # CROSSOVER
            random_number = random.random() # a random number between 0 and 1.
            if random_number < CROSSOVER_RATE:
                parent_chromosome1 = sorted(
                    random.choices(population, k=tournament_size) #selects a parent chromosome 
                )[0]

                parent_chromosome2 = sorted(
                    random.choices(population, k=tournament_size) #another parent chromosome
                )[0]

                point = random.randint(0, lenCities - 1) #a random integer representing the crossover point.

                child_chromosome1 = parent_chromosome1[1][0:point]
                for j in parent_chromosome2[1]:
                    if (j in child_chromosome1) == False:
                        child_chromosome1.append(j)

                child_chromosome2 = parent_chromosome2[1][0:point]
                for j in parent_chromosome1[1]:
                    if (j in child_chromosome2) == False:
                        child_chromosome2.append(j)

            # If crossover not happen
            else:
                child_chromosome1 = random.choices(population)[0][1]
                child_chromosome2 = random.choices(population)[0][1]

            # MUTATION
            if random.random() < MUTATION_RATE:
                point1 = random.randint(0, lenCities - 1) #the first mutation point.
                point2 = random.randint(0, lenCities - 1) #the second mutation point.
                child_chromosome1[point1], child_chromosome1[point2] = (
                    child_chromosome1[point2],
                    child_chromosome1[point1], #swaps the cities at the mutation points
                )

                point1 = random.randint(0, lenCities - 1) #the first mutation point
                point2 = random.randint(0, lenCities - 1) #the second mutation point
                child_chromosome2[point1], child_chromosome2[point2] = (
                    child_chromosome2[point2],
                    child_chromosome2[point1], #swaps the cities at the mutation points
                )

            new_population.append([calcDistance(child_chromosome1), child_chromosome1])
            new_population.append([calcDistance(child_chromosome2), child_chromosome2])

        population = new_population

        gen_number += 1

        if gen_number % 5 == 0:
            fittest_chromosome = sorted(population)[0]
            fittest_distance = round(fittest_chromosome[0], 2)
            fittest_city_order = [city[0] for city in fittest_chromosome[1]]
            print("Current generation:", gen_number)
            print("Lowest path to travel so far:", fittest_distance)
            print("Best city order:", fittest_city_order)
            print("******************************************************")

        

    answer = sorted(population)[0]

    return answer, gen_number


# draw cities and answer map
def drawMap(city, answer): 
    for j in city:
        plt.plot(j[1], j[2], "ro") #plot a red dot representing the city's coordinates on the graph.
        plt.annotate(j[0], (j[1], j[2])) #annotate the city with its label at the specified coordinates.

    for i in range(len(answer[1])):
        try:
            first = answer[1][i]
            secend = answer[1][i + 1]

            plt.plot([first[1], secend[1]], [first[2], secend[2]], 'o-g', ms = 20 , mfc = 'hotpink', mec = 'hotpink')
        except:
            continue

    first = answer[1][0]
    secend = answer[1][-1]
    plt.plot([first[1], secend[1]], [first[2], secend[2]], 'o-g', ms = 20 , mfc = 'hotpink', mec = 'hotpink') 
    plt.title("TSP with GA")
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.show()


def main():
    #initial values
    POPULATION_SIZE = 500
    tournament_size = 4
    MUTATION_RATE = 0.12
    CROSSOVER_RATE = 0.85

    cities = getUserCities() #Get city information from user and save it in the cities.
    firstPopulation, firstFitest = selectPopulation(cities, POPULATION_SIZE) #generate the initial population
    answer, genNumber = geneticAlgorithm( #answer: the fittest chromosome found.
        firstPopulation,
        len(cities),
        tournament_size,
        MUTATION_RATE,
        CROSSOVER_RATE,
    )

  
    print("The last generation: " + str(genNumber))
    print("Lowest path to travel: ", round(answer[0], 2))
   

    drawMap(cities, answer)


main()