import unittest
import random
from CC4.solution import challenger_finder


class CC4Tests(unittest.TestCase):
    def test_basic(self):

        test = [5, 1, 3, 2]
        k = 1
        expected = [0, 1, 1, 2]
        actual = challenger_finder(test, k)
        self.assertEqual(expected, actual)

        test = [40, 22, 30, 20]
        k = 5
        expected = [0, 1, 0, 1]
        actual = challenger_finder(test, k)
        self.assertEqual(expected, actual)

    def test_cover_whole(self):
        # Each k large enough to cover all list
        list1 = [-10, 10, 1, 2, 100]
        k = 2000
        expected = [len(list1) - 1] * len(list1)
        actual = challenger_finder(list1, k)
        self.assertEqual(expected, actual)

        list2 = [1]
        k = 0
        expected = [0]
        actual = challenger_finder(list2, k)
        self.assertEqual(expected, actual)

        list3 = [i % 5 for i in range(100)]
        k = 10
        expected = [99] * 100
        actual = challenger_finder(list3, k)
        self.assertEqual(expected, actual)

    def test_cover_only_one(self):
        list1 = [50, 30, 10, 25]
        k = 3
        expected = [0] * 4
        actual = challenger_finder(list1, k)
        self.assertEqual(expected, actual)

        list2 = [331, 325, 231, 232, 335, 320]
        k = 0
        expected = [0] * len(list2)
        actual = challenger_finder(list2, k)
        self.assertEqual(expected, actual)

        list3 = [-20, 10, -10, 22, 0]
        k = 5
        expected = [0] * len(list3)
        actual = challenger_finder(list3, k)
        self.assertEqual(expected, actual)

    def test_comprehensive_small(self):
        random.seed(331)

        list1 = [random.randint(-100, 100) for _ in range(20)]
        k = 20
        actual = challenger_finder(list1, k)
        expected = [6, 2, 2, 2, 6, 1, 7, 3, 3, 5, 8, 2, 4, 7, 5, 4, 8, 3, 4, 8]
        self.assertEqual(expected, actual)

        list2 = [10, 20, 10, 13, 7, 21]
        k = 3
        actual = challenger_finder(list2, k)
        expected = [3, 1, 3, 2, 2, 1]
        self.assertEqual(expected, actual)

        list3 = [random.randint(-100, 100) for _ in range(20)]
        list3 = list(set(list3))
        k = 0
        actual = challenger_finder(list3, k)
        expected = [0] * len(list3)
        self.assertEqual(expected, actual)

    def test_comprehensive_large(self):
        random.seed(20202563)

        list1 = [random.randint(-100, 100) for _ in range(1000)]
        k = 250
        actual = challenger_finder(list1, k)
        expected = [999] * 1000
        self.assertEqual(expected, actual)

        list2 = [random.randint(-2000, 2000) for _ in range(1000)]
        k = 1331
        actual = challenger_finder(list2, k)
        expected = [476, 655, 677, 621, 424, 533, 668, 664, 418, 559,
                    470, 528, 379, 612, 437, 666, 672, 637, 568, 682,
                    660, 347, 659, 668, 595, 663, 678, 659, 384, 581, 367, 659, 545, 542, 657, 330, 371, 678, 374, 664, 343, 478, 668, 507, 661, 428, 411, 345, 678, 476, 680, 672, 443, 489, 666, 652, 667, 633, 568, 660, 531, 662, 342, 509, 659, 662, 634, 389, 613, 495, 622, 603, 676, 653, 667, 681, 427, 605, 665, 666, 664, 538, 677, 538, 400, 524, 680, 592, 629, 503, 453, 643, 570, 355, 681, 420, 478, 364, 581, 344, 656, 668, 366, 683, 401, 556, 668, 643, 586, 330, 422, 663, 682, 513, 670, 490, 647, 472, 533, 502, 355, 465, 592, 667, 663, 664, 659, 672, 633, 494, 591, 450, 384, 667, 666, 661, 423, 333, 672, 674, 676, 663, 633, 369, 609, 665, 684, 419, 666, 653, 677, 677, 658, 365, 658, 666, 628, 443, 660, 355, 652, 665, 407, 629, 530, 658, 662, 685, 663, 358, 427, 667, 678, 664, 623, 440, 634, 486, 419, 666, 607, 340, 661, 680, 388, 679, 681, 658, 662, 419, 661, 672, 663, 680, 346, 525, 680, 523, 387, 680, 659, 483, 667, 373, 358, 374, 667, 659, 546, 667, 551, 487, 620, 662, 655, 442, 668, 368, 616, 661, 629, 339, 370, 685, 659, 666, 660, 683, 672, 447, 663, 420, 374, 490, 666, 659, 521, 365, 661, 668, 512, 418, 668, 609, 364, 530, 378, 682, 399, 367, 419, 409, 467, 659, 663, 335, 385, 340, 372, 514, 574, 679, 571, 393, 685, 379, 619, 454, 358, 478, 647, 448, 667, 387, 528, 374, 664, 409, 666, 598, 670, 596, 451, 668, 583, 668, 654, 563, 595, 465, 521, 661, 439, 330, 514, 581, 483, 660, 668, 477, 345, 683, 626, 659, 465, 631, 596, 409, 395, 466, 659, 330, 666, 521, 669, 376, 669, 575, 663, 638, 596, 340, 665, 443, 666, 382, 352, 671, 665, 659, 437, 581, 659, 652, 570, 528, 654, 579, 443, 666, 359, 351, 562, 512, 670, 579, 662, 654, 668, 661, 663, 658, 660, 419, 468, 460, 562, 541, 638, 503, 671, 673, 372, 666, 668, 679, 386, 663, 366, 633, 681, 374, 647, 665, 652, 420, 576, 409, 353, 662, 578, 546, 339, 460, 465, 351, 544, 617, 420, 340, 455, 365, 467, 370, 683, 435, 639, 680, 666, 457, 469, 608, 418, 349, 446, 589, 626, 446, 660, 662, 667, 624, 365, 665, 554, 661, 641, 646, 658, 635, 561, 662, 408, 662, 452, 683, 317, 337, 344, 494, 497, 467, 660, 676, 669, 628, 494, 358, 419, 484, 541, 668, 419, 667, 340, 664, 657, 533, 530, 447, 457, 598, 403, 667, 436, 671, 400, 659, 447, 489, 599, 442, 383, 661, 529, 424, 679, 555, 377, 678, 424, 467, 661, 662, 656, 677, 435, 520, 409, 676, 378, 523, 634, 677, 403, 547, 676, 621, 458, 476, 452, 658, 333, 533, 568, 530, 661, 631, 664, 465, 663, 662, 632, 355, 659, 489, 624, 586, 389, 370, 409, 654, 449, 320, 661, 668, 520, 365, 437, 662, 368, 565, 374, 589, 377, 346, 633, 446, 680, 420, 668, 467, 661, 661, 456, 681, 669, 592, 368, 410, 680, 653, 629, 556, 571, 666, 667, 419, 571, 640, 451, 360, 685, 663, 663, 340, 662, 494, 580, 570, 669, 679, 425, 430, 514, 668, 556, 659, 559, 666, 671, 668, 419, 447, 385, 580, 653, 565, 664, 538, 665, 484, 676, 503, 598, 666, 659, 662, 402, 538, 441, 664, 667, 420, 653, 678, 633, 660, 363, 681, 663, 656, 668, 652, 656, 436, 576, 661, 561, 684, 669, 401, 345, 669, 659, 430, 656, 534, 613, 509, 562, 579, 420, 677, 592, 330, 523, 419, 667, 660, 667, 529, 406, 579, 672, 658, 668, 677, 554, 666, 601, 572, 665, 576, 580, 679, 314, 670, 513, 671, 662, 549, 508, 348, 330, 532, 648, 409, 573, 679, 517, 447, 589, 676, 629, 395, 533, 678, 652, 412, 561, 533, 520, 542, 320, 666, 464, 340, 593, 633, 596, 619, 681, 474, 666, 333, 465, 662, 509, 444, 509, 342, 624, 523, 670, 667, 673, 605, 443, 679, 405, 659, 535, 545, 680, 668, 361, 666, 662, 540, 400, 377, 668, 568, 665, 684, 367, 665, 365, 620, 556, 384, 638, 668, 531, 501, 372, 666, 671, 554, 427, 676, 637, 677, 566, 657, 588, 334, 451, 525, 666, 460, 555, 659, 319, 512, 657, 511, 668, 673, 665, 661, 514, 680, 647, 680, 409, 656, 677, 662, 389, 563, 579, 417, 423, 677, 643, 680, 663, 419, 659, 661, 438, 494, 663, 422, 684, 357, 403, 320, 463, 452, 647, 668, 472, 683, 404, 538, 581, 455, 427, 659, 365, 370, 641, 659, 661, 400, 541, 661, 666, 366, 665, 504, 384, 579, 670, 534, 650, 630, 637, 624, 404, 319, 680, 489, 668, 435, 648, 677, 633, 578, 680, 680, 513, 504, 353, 658, 677, 355, 508, 555, 406, 514, 670, 664, 516, 659, 676, 472, 359, 665, 681, 666, 643, 668, 370, 668, 659, 626, 679, 684, 455, 420, 564, 619, 491, 670, 420, 412, 632, 404, 633, 409, 520, 663, 666, 404, 482, 629, 466, 676, 504, 379, 664, 613, 340, 676, 481, 666, 668, 659, 610, 664, 633, 672, 591, 349, 654, 465, 342, 668, 555, 668, 655, 442, 418, 418, 469, 665, 367, 487, 561, 497, 641, 680, 571, 681, 660, 340, 369, 662, 475, 684, 574, 372, 542, 668, 523, 369, 541, 647, 654, 409, 566, 671, 523, 365, 386, 672, 677, 667, 604, 656, 588, 562, 664, 677, 583, 470, 345, 343, 659, 467, 415, 668, 677, 457, 384, 554, 662, 678, 600, 504, 465, 663, 464, 685, 633, 578, 418, 667, 593, 529, 377, 682, 610, 666, 632, 647, 668, 657, 554, 647, 659, 529, 662, 468, 598, 378, 630, 666, 665, 623, 667, 374, 664, 336, 652, 669, 555, 609, 677, 586, 531, 420, 443, 653, 614, 643]
        self.assertEqual(expected, actual)

        list3 = list(set([random.randint(-10000, 10000) for _ in range(5000)]))
        k = 0
        actual = challenger_finder(list3, k)
        expected = [0] * len(list3)
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
