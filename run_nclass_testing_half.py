import argparse

from brats_nclass_testing_half import train_isensee2017, plot_loss, predict_val_test, test_external, plot_fm, evaluate, plot_results, test_partial
# from fm import keras_vis

parser = argparse.ArgumentParser(description='Parent code for running preprocessing/training/prediction/evaluation using U-Net on GBM/METS')
parser.add_argument('-o','--op', nargs='?', help='Choose from train(training), predict(prediction), eval(evaluation), preproc(preprocessing)', default="train")
parser.add_argument('-m','--mode', nargs='?', help='Set this to True for debug mode', default=None)
parser.add_argument('-v','--val', nargs='?', help='Choose validation fold', default="1")
parser.add_argument('-e','--exp', nargs='?', help='Give experiment name', default="dev")

args = vars(parser.parse_args())

print(args)

if args['op'] == "train":
	print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Train ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
	train_isensee2017.main(fold=args['val'], exp=args['exp'], debugmode = args['mode'])

elif args['op'] == "plot":
	print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Plot loss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
	plot_loss.main(fold=args['val'], exp=args['exp'])

elif args['op'] == "predict":
	print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Predict ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
	predict_val_test.main(fold=args['val'], exp=args['exp'])

elif args['op'] == "eval":
	print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Predict ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
	evaluate.main(fold=args['val'], exp=args['exp'])

elif args['op'] == "pp":
	print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Plot + Predict ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
	plot_loss.main(fold=args['val'], exp=args['exp'])
	predict_val_test.main(fold=args['val'], exp=args['exp'])

elif args['op'] == "test":
	print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ External testing ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
	test_external.main(fold=args['val'], exp=args['exp'])

elif args['op'] == "partial":
	print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Testing partial cases ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
	test_partial.main(fold=args['val'], exp=args['exp'])

elif args['op'] == "results":
	print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Plot results ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
	plot_results.main(fold=args['val'], exp=args['exp'], cohort_suffix = 'val')
	plot_results.main(fold=args['val'], exp=args['exp'], cohort_suffix = 'test')

elif args['op'] == "defm":
	print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FM Visualization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
	plot_fm.main(fold=args['val'], exp=args['exp'])
	
