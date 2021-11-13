from dream import Dream

if __name__ == '__main__':
    dream_sim = Dream(param_file="Data/Par.csv")
    dream_sim.read_input("Data/Data.csv")
    dream_sim.simulate()
    dream_sim.plot()
