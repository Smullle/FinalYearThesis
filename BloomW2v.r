# System usage

knowledge <- c(0.2325792,0.11560252,-0.060916126,0.2325792,0.085342705,-0.01956962,0.11560252,0.085342705,0.6704947,-0.060916126,-0.01956962,0.6704947)
understanding <- c(0.8100976,0.81740427,0.5808524,0.8100976,0.68570817,0.67724836,0.81740427,0.68570817,0.42890376,0.5808524,0.67724836,0.42890376)
application <- c(0.7077052,0.75883245,0.6225083,0.404217,0.7077052,0.5935151,0.5079297,0.20354526,0.75883245,0.5935151,0.67745584,0.30699947,0.6225083,0.5079297,0.67745584,0.2901032,0.404217,0.20354526,0.30699947,0.2901032)
analysis <- c(0.47623488,0.68016946,0.23650232,0.6398237,0.47623488,0.2715749,0.24329235,0.24756946,0.68016946,0.2715749,0.37781698,0.7323598,0.23650232,0.24329235,0.37781698,0.28138226,0.6398237,0.24756946,0.7323598,0.28138226)
evaluation <- c(0.37203455,0.37182376,0.08056219,0.46184954,0.37203455,0.47663152,0.021924132,0.564018,0.37182376,0.47663152,0.106906794,0.31111228,0.08056219,0.021924132,0.106906794,-0.0066788383,0.46184954,0.564018,0.31111228,-0.0066788383)
create <- c(0.5037873,0.7953428,0.46344694,0.552953,0.5037873,0.58302367,0.15404108,0.5282043,0.7953428,0.58302367,0.41238964,0.585173,0.46344694,0.15404108,0.41238964,0.13231081,0.552953,0.5282043,0.585173,0.13231081)



boxplot(knowledge, understanding, application, analysis, evaluation, create, names=c("knowledge", "understanding", "application", "analysis", "evaluation", "create"), las=2, notch=TRUE, ylim=c(0,1.2) , cex.axis=0.6)



points(1:6, c(mean(knowledge), mean(understanding), mean(application), mean(analysis), mean(evaluation), mean(create)), pch=18 , cex=1.5)