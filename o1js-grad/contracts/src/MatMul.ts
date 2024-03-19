import { Field } from "o1js"

export const matmul = (
    A: Field[][],
    B: Field[][],
) => {
    Field(A[0].length).assertEquals(Field(B.length), "Matrix dimensions do not match for multiplication.");

    const result: Field[][] = Array(A.length).fill(0).map(() => Array(B[0].length).fill(Field(0)));

    for (let i = 0; i < A.length; i++) {
        for (let j = 0; j < B[0].length; j++) {
            let sum = Field(0);
            for (let k = 0; k < A[0].length; k++) {
                sum = sum.add(A[i][k].mul(B[k][j]));
            }
            result[i][j] = sum;
        }
    }

    return result;
}