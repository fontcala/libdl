/** @file NetworkElement.h
 *  @author Adria Font Calvarons
 */
#ifndef NETWORKELEMENT_H
#define NETWORKELEMENT_H

#include <memory>

/**
@class NetworkElement
@brief Basic interface for any element of a network
 */

#include "dlfunctions.h"
#include "dltypes.h"
template <typename DataType>
class NetworkElement
{
public:
    /**
    * NetworkElement::ForwardPass
    * 
    * A Forward pass uses the data reference obtained during a \c SetInput call and 
    * sets the output value which is may be used by another NetworkElement object.
    */
    virtual void ForwardPass() = 0;

    /**
    * NetworkElement::BackwardPass
    * 
    * A Backward pass uses the data reference obtained during a \c SetBackpropInput call and 
    * sets the bacpropagation value which is may be used by another NetworkElement object.
    */
    virtual void BackwardPass() = 0;

    // Helpers to connect Layers
    virtual void SetData(const Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput) = 0;
    virtual void SetInput(const Eigen::Matrix<DataType, Dynamic, Dynamic> *aInput) = 0;
    virtual void SetBackpropInput(const Eigen::Matrix<DataType, Dynamic, Dynamic> *aOutput) = 0;
    virtual const Eigen::Matrix<DataType, Dynamic, Dynamic> *GetOutput() const = 0;
    virtual const Eigen::Matrix<DataType, Dynamic, Dynamic> *GetBackpropOutput() const = 0;
};

#endif