\documentclass{article}
\usepackage{tikz}
\usetikzlibrary{shapes, arrows, positioning}

\begin{document}

\begin{figure}
\centering
\begin{tikzpicture}[
    block/.style={rectangle, draw, fill=blue!30, text width=7em, text centered, rounded corners, minimum height=2em},
    decision/.style={diamond, draw, fill=blue!20, text width=6em, text centered, minimum height=4em, aspect=2},
    arrow/.style={thick,->,>=stealth}
]

% Nodes
\node[block] (query) {User Query};
\node[block, below=0.5cm of query] (bert) {BERT Classifier};
\node[decision, below=0.5cm of bert] (route) {Route Decision};

% Branch A (Left)
\node[block, below left=1.0cm and 1.5cm of route] (vdb) {Vector DB Retrieval};
\node[block, below=0.5cm of vdb] (llm) {Fine-tuned LLM};

% Branch B/C (Right)
\node[block, below right=1.0cm and 1.5cm of route] (safety) {Safety Filter};

% Response (Shared)
\node[block, below=2.5cm of route] (response) {Response};

% Connect the main pipeline
\draw[arrow] (query) -- (bert);
\draw[arrow] (bert) -- (route);

% Connect the Route Decision branches
\draw[arrow] (route.west) -- node[above, near start] {Type A} (vdb);
\draw[arrow] (route.east) -- node[above, near start] {Type B/C} (safety);

% Connect Branch A to Response
\draw[arrow] (vdb) -- (llm);
\draw[arrow] (llm) -- (response);

% Connect Branch B/C to Response
\draw[arrow] (safety) -- (response);

\end{tikzpicture}
\caption{System Architecture Pipeline}
\end{figure}

\end{document}
