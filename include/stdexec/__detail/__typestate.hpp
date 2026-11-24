/*
 * Copyright (c) 2026 NVIDIA Corporation
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "__config.hpp"

#if STDEXEC_USE_MODULES() && !defined(STDEXEC_IN_MODULE_PURVIEW)

import stdexec;

#else

#  include "__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#  include "__concepts.hpp"
#  include "__meta.hpp"
#  include "__optional.hpp"
#  include "__sender_facade.hpp"
#  include "__tuple.hpp"
#  include "__type_traits.hpp"
#  include "__utility.hpp"
#  include "__variant.hpp"

#  if !STDEXEC_USE_MODULES()
#    include <cstddef>
#    include <exception>
#  endif

#  include "__prologue.hpp"

// Operation states as phases: a way to write the state of a sender facade algorithm (see
// __sender_facade.hpp) whose nested operations do not all live as long as the algorithm's
// operation. See MAINTAINERS.md for a guide.
//
// The state of such an algorithm has two kinds of member:
//
//   1. Things that live as long as the operation: the receiver, the algorithm's data, the
//      senders it connects later, and operations connected when the operation is (such as
//      continues_on's schedule operation). These are ordinary members.
//
//   2. Things that live only while the operation is in a particular phase. A *phase* is a
//      nested operation that runs for a while and is then destroyed: let_value's child,
//      and then the sender that its function returns; each of __sequence's senders in
//      turn; each iteration of exec::repeat_n's body. The phases that cannot be live at
//      the same time share a region of the state, a __phases<...>. Each phase is named by
//      the facade key of its operation, so the key of a completion says which phase it
//      came from, and __complete(key, ...) handles the completions of that phase.
//
// Values that outlive the phase they come from, such as the results that let_value passes
// to its function, are received into a region of their own while the phase is still live,
// because they may be in that phase's operation.
//
// The verbs here are the only ways in which an algorithm enters, starts and ends phases.
// Each is written once with the lifetime rules that the hand-written states repeated:
//
//   - __run<Key>(phases, connect): end the live phase, then connect phase Key's operation
//     and start it. (The live phase is the one whose completion is being handled: an
//     operation that has completed may be destroyed. Anything that must outlive it was
//     received, or copied, before this.)
//   - __start<Key>(phases): start the operation of phase Key, entered earlier (when the
//     operation was connected, say).
//   - __catching(rcvr, fn, on_throw): do fn, which can throw. If it does, do on_throw (which
//     ends any phases that must end before the operation completes), and complete with
//     the exception as an error.
//   - __receive(storage, tag, args...) and __send_received(storage, rcvr): save a
//     completion (which may be in a phase's operation), and send it later.
//   - __finish_with_copies(rcvr, end, tag, args...): copy a completion's arguments (which
//     may be in a phase's operation), end the phases, and send the copies.
//
// Each verb returns the facade's effect token for what it did, so the facade derives the
// algorithm's completions from the verbs as from any other code.

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  namespace __typestate
  {
    ////////////////////////////////////////////////////////////////////////////////////////
    // A phase: the nested operation keyed by _Key (a facade key, such as __child<0>), whose
    // operation state has type _OpState
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Key, class _OpState>
    struct __phase
    {
      using __key_t = _Key;
      using __op_t  = _OpState;
    };

    template <class _Key, class... _Phases>
    inline constexpr std::size_t __phase_index_v =
      STDEXEC::__index_of<_Key, typename _Phases::__key_t...>();

    ////////////////////////////////////////////////////////////////////////////////////////
    // A region of an operation's state in which the phases _Phases live, one at a time. At
    // first no phase is live. (A region of one phase is an __optional, and a region of
    // several is a __variant, as the hand-written states had them.)
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class... _Phases>
    struct __phases
    {
      template <class _Key>
      static constexpr std::size_t __index_v = __phase_index_v<_Key, _Phases...>;

      template <class _Key>
      using __op_t = __m_at_c<__index_v<_Key>, typename _Phases::__op_t...>;

      __phases() = default;
      STDEXEC_IMMOVABLE(__phases);

      // Is phase _Key live?
      template <class _Key>
      [[nodiscard]] STDEXEC_ATTRIBUTE(always_inline,
                                      host,
                                      device) constexpr auto __is_live() const noexcept -> bool
      {
        return __ops_.index() == __index_v<_Key>;
      }

      // The operation of phase _Key, which must be live
      template <class _Key>
      [[nodiscard]] STDEXEC_ATTRIBUTE(always_inline, host, device) constexpr auto __op() noexcept
        -> __op_t<_Key> &
      {
        return __var::__get<__index_v<_Key>>(__ops_);
      }

      // Ends the live phase, if there is one, and enters phase _Key: its operation is the
      // result of __connect(). If that throws, no phase is live.
      template <class _Key, class _Connect>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      constexpr auto __enter(_Connect &&__connect) noexcept(__nothrow_callable<_Connect>)
        -> __op_t<_Key> &
      {
        static_assert(__same_as<__call_result_t<_Connect>, __op_t<_Key>>,
                      "__connect must return the operation state of the phase");
        return __ops_.template __emplace_from<__index_v<_Key>>(static_cast<_Connect &&>(__connect));
      }

     private:
      __variant<typename _Phases::__op_t...> __ops_{__no_init};
    };

    // A region with no phases, as of a __sequence of no senders
    template <>
    struct __phases<>
    {};

    // A region of one phase can also end that phase without entering another, for an
    // algorithm that destroys its nested operations before it completes (exec::repeat_n).
    template <class _Phase>
    struct __phases<_Phase>
    {
      template <class _Key>
      static constexpr std::size_t __index_v = __phase_index_v<_Key, _Phase>;

      template <class _Key>
      using __op_t = _Phase::__op_t;

      __phases() = default;
      STDEXEC_IMMOVABLE(__phases);

      template <class _Key>
      [[nodiscard]] STDEXEC_ATTRIBUTE(always_inline,
                                      host,
                                      device) constexpr auto __is_live() const noexcept -> bool
      {
        static_assert(__index_v<_Key> == 0);
        return __op_.has_value();
      }

      template <class _Key>
      [[nodiscard]] STDEXEC_ATTRIBUTE(always_inline, host, device) constexpr auto __op() noexcept
        -> __op_t<_Key> &
      {
        static_assert(__index_v<_Key> == 0);
        STDEXEC_ASSERT(__op_.has_value());
        return *__op_;
      }

      template <class _Key, class _Connect>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      constexpr auto __enter(_Connect &&__connect) noexcept(__nothrow_callable<_Connect>)
        -> __op_t<_Key> &
      {
        static_assert(__index_v<_Key> == 0);
        static_assert(__same_as<__call_result_t<_Connect>, __op_t<_Key>>,
                      "__connect must return the operation state of the phase");
        return __op_.__emplace_from(static_cast<_Connect &&>(__connect));
      }

      STDEXEC_ATTRIBUTE(always_inline, host, device)
      constexpr void __end() noexcept
      {
        __op_.reset();
      }

     private:
      __optional<typename _Phase::__op_t> __op_{};
    };

    ////////////////////////////////////////////////////////////////////////////////////////
    // Verbs. Each returns the facade's token for what it did.

    // Starts the operation of phase _Key, which must be live
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Key, class _Phases>
    STDEXEC_ATTRIBUTE(always_inline, host, device)
    constexpr auto __start(_Phases &__phases) noexcept
    {
      return __facade::__start_nested<_Key>(__phases.template __op<_Key>());
    }

    // Ends the live phase (whose completion is being handled, typically), enters phase _Key,
    // whose operation __connect() connects, and starts it. If __connect() throws, the
    // exception propagates, and no phase is live. (See __catching.)
    //
    // Nothing that is in the live phase's operation may be used after this begins: receive
    // or copy what must outlive the phase first. And nothing in the state may be used after
    // it returns, because the operation that it starts may complete inline, and the
    // algorithm's operation can be destroyed when it completes.
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Key, class _Phases, class _Connect>
    STDEXEC_ATTRIBUTE(always_inline, host, device)
    constexpr auto __run(_Phases &__phases, _Connect &&__connect)
      noexcept(__nothrow_callable<_Connect>) -> __facade::__effects<__facade::__starts<_Key>>
    {
      auto &__op = __phases.template __enter<_Key>(static_cast<_Connect &&>(__connect));
      return __facade::__start_nested<_Key>(__op);
    }

    // Does __fn(), which returns the token of what it did. If __fn() can throw and does, does
    // __on_throw() and then sends the exception to __rcvr as an error. (__on_throw ends the
    // phases that must end before the operation completes, if any.) Whether __fn() can throw
    // is its exception specification, so a lambda that cannot throw must say so, as with
    // `[&]() noexcept(__nothrow_v) { ... }`. Otherwise the operation declares an error
    // completion that never happens.
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Receiver, class _Fn, class _OnThrow = __ignore>
    STDEXEC_ATTRIBUTE(always_inline, host, device)
    constexpr auto __catching(_Receiver &__rcvr, _Fn &&__fn, _OnThrow &&__on_throw = {}) noexcept
    {
      if constexpr (__nothrow_callable<_Fn>)
      {
        return static_cast<_Fn &&>(__fn)();
      }
      else
      {
        using __result_t =
          __facade::__union_t<__call_result_t<_Fn>,
                              __facade::__effects<set_error_t(std::exception_ptr)>>;
        STDEXEC_TRY
        {
          return __result_t(static_cast<_Fn &&>(__fn)());
        }
        STDEXEC_CATCH_ALL
        {
          if constexpr (!__same_as<__decay_t<_OnThrow>, __ignore>)
          {
            static_cast<_OnThrow &&>(__on_throw)();
          }
          return __result_t(__facade::__send(__rcvr, set_error, std::current_exception()));
        }
      }
    }

    // Saves the completion _Tag(__args...) in __storage, a __results_storage, so that it can
    // be sent after the phase that it came from has ended. If decay-copying the arguments
    // throws, saves the exception as an error instead. (__results_storage has room for it.)
    // Saving is not an effect, so this returns nothing. The arguments must be
    // decay-copyable: an algorithm reports that they are not as a type error (see
    // __facade::__decay_copy_error).
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Storage, class _Tag, class... _Args>
      requires __decay_copyable<_Args...>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr void __receive(_Storage &__storage, _Tag __tag, _Args &&...__args) noexcept
    {
      if constexpr (__nothrow_decay_copyable<_Args...>)
      {
        __storage.__emplace_from(__mktuple, __tag, static_cast<_Args &&>(__args)...);
      }
      else
      {
        STDEXEC_TRY
        {
          __storage.__emplace_from(__mktuple, __tag, static_cast<_Args &&>(__args)...);
        }
        STDEXEC_CATCH_ALL
        {
          __storage.__emplace_from(__mktuple, set_error, std::current_exception());
        }
      }
    }

    // Sends the completion that __receive saved in __storage
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Storage, class _Receiver>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto __send_received(_Storage &__storage, _Receiver &__rcvr) noexcept
    {
      return __facade::__visit(
        __storage,
        [&]<class _Tuple>(_Tuple &__tupl) noexcept
        {
          return STDEXEC::__apply(
            [&]<class _Tag, class... _Args>(_Tag __tag, _Args &...__args) noexcept
            { return __facade::__send(__rcvr, __tag, static_cast<_Args &&>(__args)...); },
            __tupl);
        });
    }

    // Sends _Tag(__args...), whose arguments may be in the operation of a live phase, after
    // __end() ends the phases. The arguments are decay-copied first. If _Decay, the copies are
    // sent as rvalues; otherwise with the value categories of the arguments. If copying
    // throws, the phases are ended and the exception is sent as an error.
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <bool _Decay, class _Receiver, class _End, class _Tag, class... _Args>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto __finish_with_copies(_Receiver &__rcvr,
                                        _End       __end,
                                        _Tag       __tag,
                                        _Args &&...__args) noexcept
    {
      return __typestate::__catching(
        __rcvr,
        [&]() noexcept(__nothrow_decay_copyable<_Args...>)
        {
          __decayed_tuple<_Args...> __copies{static_cast<_Args &&>(__args)...};
          __end();
          return STDEXEC::__apply(
            [&](auto &...__copy) noexcept
            {
              if constexpr (_Decay)
                return __facade::__send(__rcvr, __tag, static_cast<__decay_t<_Args> &&>(__copy)...);
              else
                return __facade::__send(__rcvr, __tag, static_cast<_Args &&>(__copy)...);
            },
            __copies);
        },
        __end);
    }
  }  // namespace __typestate
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
