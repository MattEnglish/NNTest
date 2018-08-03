using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Neuro;
using Accord.Math;
using Accord.Statistics;
using Accord.Neuro.Learning;
using Accord.Genetic;
using Accord.Math.Random;

namespace NeuralNetTest
{

    public class learnData
    {
        public double currentDrawStreak;
        public double MyWinsLeft;
        public double EnemyWinsLeft;
        public double MyDynamiteLeft;
        public double EnemyDynamiteLeft;
    }

    public class fitFunc: IFitnessFunction
    {
        public double Evaluate(IChromosome chromosone)
        {
            var test = (DoubleArrayChromosome)chromosone;
            var value = test.Value;
            var net = new NeuralNetworks();

            for (int i = 0; i < 6; i++)
            {
                 net.W1[i/3,i%3] = value[i];
            }
            for (int i = 6; i < 9; i++)
            {
                net.W2[i -6, 0] = value[i];
            }

            var a = Math.Pow(net.Forward(new double[1, 2] { { 0, 1 } })[0,0] -1,2);
            var b = Math.Pow(net.Forward(new double[1, 2] { { 1, 0 } })[0, 0] - 1, 2);
            var c = Math.Pow(net.Forward(new double[1, 2] { { 0, 0 } })[0, 0], 2);
            var d = Math.Pow(net.Forward(new double[1, 2] { { 1, 1 } })[0, 0], 2);

            return Math.Max( 1 - a - b - c - d, 0);

        }
    }


    public class AccordTest
    {
        public static void RunGene()
        {
            var rand = new ZigguratExponentialGenerator();
            var chrom = new DoubleArrayChromosome(rand, rand, rand, 9 );
            var pop = new Population(1000, chrom, new fitFunc(), new EliteSelection() );
            Console.WriteLine(pop.FitnessMax);
            for (int i = 0; i < 199; i++)
            {
                pop.RunEpoch();
                Console.WriteLine(pop.FitnessMax);
            }
            var x = new fitFunc().Evaluate(pop.BestChromosome);

            Console.WriteLine(pop.FitnessMax);
        }


        public static void Run()
        {
            double[][] input =
            {
                new double[] {0, 0}, new double[] {0, 1},
                new double[] {1, 0}, new double[] {1, 1}
            };

            double[][] output =
            {
                new double[] {0}, new double[] {1},
                new double[] {1}, new double[] {0}
            };
            var network = new ActivationNetwork(new SigmoidFunction(2), 2, 4, 1);
            var y = network.Compute(new double[] { 1, 0 });
            var teacher = new LevenbergMarquardtLearning(network);

            
            teacher.RunEpoch(input, output);
            var z = network.Compute(new double[] { 1, 0 });

            teacher.RunEpoch(input, output);
            teacher.RunEpoch(input, output);
            teacher.RunEpoch(input, output);
            teacher.RunEpoch(input, output);
            var badsf = network.Compute(new double[] { 1, 0 });
            var test2 = network.Compute(new double[] { 1, 1 });
        }

    }
}
