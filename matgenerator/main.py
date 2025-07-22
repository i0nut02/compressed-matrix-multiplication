from matGenerator import MatrixGenerator

if __name__ == "__main__":
    generator = MatrixGenerator()

    generator.setRows(10000).setColumns(10000).setNNZPercentage(0.01).setValueRange(-100, 100).setFilename("A.inp").setPath("./matrices/").setRandomSeed(1).generate()
    generator.setRows(10000).setColumns(10000).setNNZPercentage(0.01).setValueRange(-100, 100).setFilename("B.inp").setPath("./matrices/").setRandomSeed(2).generate()
    generator.setRows(10000).setColumns(10000).setNNZPercentage(0.01).setValueRange(-100, 100).setFilename("C.inp").setPath("./matrices/").setRandomSeed(3).generate()