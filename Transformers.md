
![[Pasted image 20250924211033.png]]

How does the *ATTENTION* work?

It lets the model focus on the most relevant tokens in the input when processing a given token.

For example each word in sentence:
1. It is projected into 3 vectors:
		Query(Q)
		Key(K)
		Value(V)
   2. Computing attention scores between token's query and every other token's key.
	   Mathematically:
		   ![[Pasted image 20250924211502.png]]
	3. Apply softmax to turn scores into probabilities (attention weights).
	4. Multiply weights by the value vectors -> weighted sum gives the output for this token.

	*Example:*
		The cat sat on the mat.
			*when encoding "sat", attention might assign:
			High weight to "cat"(who sat)
			Medium weight to "mat"(where sat)
			Low weight to "the"*
		So the representation of "sat" is enriched with context from "cat" and "mat".
	