/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once

namespace MegBA {
    namespace CHK {
        template<typename T>
        inline bool Device_Same(const JetVector<T> &f, const JetVector<T> &g) {
            return f.get_Device() == g.get_Device();
        }

        template<typename T>
        inline void Device_Throw(const JetVector<T> &f, const JetVector<T> &g) {

            if (!Device_Same(f, g))
                throw std::runtime_error(
                        "Different device_ for item #1 is on " + std::to_string(f.get_Device()) + " item #2 is on " +
                        std::to_string(g.get_Device()));
        }

        template<typename T>
        inline bool Shape_Same(const JetVector<T> &f, const JetVector<T> &g) {
            const auto f_Grad_Shape = f.get_Grad_Shape();
            const auto g_Grad_Shape = g.get_Grad_Shape();
            return (f_Grad_Shape == 0 || g_Grad_Shape == 0 || f_Grad_Shape == g_Grad_Shape) && f.get_Elm_Num() == g.get_Elm_Num() ? true : false;
        }

        template<typename T>
        inline void Shape_Throw(const JetVector<T> &f, const JetVector<T> &g) {
            if (!Shape_Same(f, g))
                throw std::runtime_error(
                        "Different shape for gradient of item #1 is " + std::to_string(f.get_Grad_Shape()) +
                        " item #2 is " + std::to_string(g.get_Grad_Shape()) +
                        ", element number of item #1 is " + std::to_string(f.get_Elm_Num()) +
                        " item #2 is " + std::to_string(g.get_Elm_Num()));
        }
    }
}

