import { Field } from 'o1js';
import { matmul } from './MatMul';

describe('matmul function', () => {
    it('computes matrix-matrix multiplication correctly', () => {
        const A = [
            [new Field(1), new Field(2)],
            [new Field(3), new Field(4)]
        ];
        const B = [
            [new Field(2), new Field(0)],
            [new Field(1), new Field(2)]
        ];
        const result = matmul(A, B);
        const expected = [
            [new Field(4), new Field(4)], // (1*2 + 2*1), (1*0 + 2*2)
            [new Field(10), new Field(8)] // (3*2 + 4*1), (3*0 + 4*2)
        ];
        expect(result).toEqual(expected);
    });

    it('throws error on dimension mismatch', () => {
        const A = [
            [new Field(1), new Field(2), new Field(3)]
        ];
        const B = [
            [new Field(1)],
            [new Field(2)]
            // Missing third row to match A's columns
        ];
        expect(() => matmul(A, B)).toThrow("Matrix dimensions do not match for multiplication.");
    });
});
