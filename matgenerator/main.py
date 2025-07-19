from matGenerator import MatrixGenerator

if __name__ == "__main__":
    generator = MatrixGenerator()

    generator.setRows(10000).setColumns(12000).setNNZPercentage(0.01).setValueRange(-100, 100).setFilename("A.txt").setPath("./matrices/").setRandomSeed(1).generate()
    generator.setRows(12000).setColumns(10000).setNNZPercentage(0.01).setValueRange(-100, 100).setFilename("B.txt").setPath("./matrices/").setRandomSeed(2).generate()