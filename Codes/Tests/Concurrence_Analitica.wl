(* Concurrence_Analitica.wl - versione script autoconsistente *)

ClearAll[ConcurrenceWootters, ConcurrenceAnalytical, sigmaY, rhoExample, ass, concExample];

sigmaY = {{0, -I}, {I, 0}};

ConcurrenceWootters[rho_] := Module[{YY, rhoTilde, ev, lambdas},
  YY = KroneckerProduct[sigmaY, sigmaY];
  rhoTilde = YY . Conjugate[rho] . YY;
  ev = Eigenvalues[rho . rhoTilde];
  lambdas = Reverse@Sort[Re@Sqrt[Chop[ev]]];
  Max[0, lambdas[[1]] - Total[lambdas[[2 ;; 4]]]]
];

ConcurrenceAnalytical[rho_, ass_: True] := FullSimplify[ConcurrenceWootters[rho], Assumptions -> ass];

rhoExample = 1/2 {{1, 0, 0, -Exp[-2 (1 - eta) gamma t]},
                 {0, 0, 0, 0},
                 {0, 0, 0, 0},
                 {-Exp[-2 (1 - eta) gamma t], 0, 0, 1}};

ass = Element[{gamma, eta, t}, Reals] && gamma > 0 && eta > 0 && t >= 0 && eta <= 1;
concExample = ConcurrenceAnalytical[rhoExample, ass];

Print["Concurrence analitica = ", concExample];
