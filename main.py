import dataProc.visual
from dataProc.visual import *
from dataGeneration.generator import *
from dataGeneration.format import *


def generate():
    print(">>> Running")
    start = cubic_space(start=0.5, shift_x=0, shift_y=0, cube_size=1, cube_factor=1)

    print(">>> Generating calculation data")
    generation_data = acquire_data(
        start,
        iterations=250,
        use_json=False,
        generate_from_angles=False,
        angles={'t_start': 0, 't_end': 90, 'p_start': 0, 'p_end': 90, 'dt': 5, 'dp': 5},
        from_cubic_space={'start': 0, 'cube_factor': 1, 'cube_size': 2}
    )

    # print the generation data into a json file
    print(">>> Saving generated data")
    to_json({'data': generation_data}, 'generation_data')

    # Plot the trajectory vectors and the starting points
    plot_3d_vectors(list(set([d['trajectory'] for d in generation_data])))
    plot_3d_vectors(list(set([d['start'] for d in generation_data])))

    # calculate the delta sequences for the numeric cmf and pcf limit calculations of the cmf
    print(">>> calculating ...")
    data = generate_delta_sequence(generation_data, cmf_deltas=True, pcf_deltas=True)
    data = generate_convergence(data, cmf_conv=True, pcf_conv=True)

    # print the calculated data into a json file
    print(">>> Saving results")
    to_json({'data': data}, 'data')


def main():
    data = from_json('./dataGeneration/data')
    formatted: Dict[Tuple] = unpack_json(data)

    snip_2d_compare(formatted,
                    start=(0.5, 0.5, 0.5),
                    trajectory=(1, 1, 0),
                    data_label_a='deltas_unknown_pcf_limit',
                    data_label_b='deltas_unknown_cmf_limit', alpha=0.6)


if __name__ == '__main__':
    main()
