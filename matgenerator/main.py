from matGenerator import MatrixGenerator

if __name__ == "__main__":
    generator = MatrixGenerator()

    #generator.setRows(10000).setColumns(10000).setNNZPercentage(0.01).setValueRange(-100, 100).setFilename("A.inp").setPath("../matrices/").setRandomSeed(1).generate()
    #generator.setRows(20000).setColumns(20000).setNNZPercentage(0.01).setValueRange(-100, 100).setFilename("B.inp").setPath("../matrices/").setRandomSeed(2).generate()
    #generator.setRows(5000).setColumns(5000).setNNZPercentage(0.01).setValueRange(-100, 100).setFilename("C.inp").setPath("../matrices/").setRandomSeed(3).generate()
    #generator.setRows(1000).setColumns(1000).setNNZPercentage(0.01).setValueRange(-100, 100).setFilename("D.inp").setPath("../matrices/").setRandomSeed(4).generate()
    generator.setRows(100).setColumns(100).setNNZPercentage(0.01).setValueRange(-100, 100).setFilename("A.inp").setPath("../matrices/").setRandomSeed(1).generate()
